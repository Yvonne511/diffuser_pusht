import torch

def pusht_data_augmentation(obs, state_normalizer=None, acts=None, action_normalizer=None):
    # seed everything
    # torch.manual_seed(0)
    device = obs.device
    states = obs
    B, T = states.shape[0], states.shape[1]

    # states = state_normalizer.to(device).unnormalize(states)  # B, T, 8
    if acts is not None:
        acts = acts.view(B, -1, 2)
    # acts = action_normalizer.to(device).unnormalize(acts)

    dx_min, dx_max, dy_min, dy_max = get_max_dx_dy(states)

    probs = torch.tensor([0.4, 0.3, 0.3, 0.0], device=states.device)
    aug_type_idx = torch.multinomial(probs, num_samples=1, replacement=True).item()
    aug_types = ['displace', 'rotate', 'displace_rotate', 'static']
    aug_type = aug_types[aug_type_idx]

    if aug_type == 'displace':
        dx = torch.stack([torch.randint(int(low.item()), int(high.item()) + 1, (1,), device=states.device)
                        for low, high in zip(dx_min, dx_max)]).squeeze(1)
        dy = torch.stack([torch.randint(int(low.item()), int(high.item()) + 1, (1,), device=states.device)
                        for low, high in zip(dy_min, dy_max)]).squeeze(1)
        theta = torch.zeros(B, device=states.device)  # No rotation

    elif aug_type == 'rotate':
        theta = torch.deg2rad(360 * torch.rand(B, device=states.device))
        dx = torch.zeros(B, device=states.device)
        dy = torch.zeros(B, device=states.device)

    elif aug_type == 'displace_rotate':
        dx_range_half = (dx_max - dx_min) // 2 + 1
        dy_range_half = (dy_max - dy_min) // 2 + 1

        dx = torch.stack([torch.randint(int(low.item()), int((low + half).item()) + 1, (1,), device=states.device)
                        for low, half in zip(dx_min, dx_range_half)]).squeeze(1)
        dy = torch.stack([torch.randint(int(low.item()), int((low + half).item()) + 1, (1,), device=states.device)
                        for low, half in zip(dy_min, dy_range_half)]).squeeze(1)
        theta = torch.deg2rad(30 * torch.rand(B, device=states.device))
    else:
        dx = torch.zeros(B, device=states.device)
        dy = torch.zeros(B, device=states.device)
        theta = torch.zeros(B, device=states.device)

    augmented_states, augmented_acts = rotate_and_displace(states, theta, dx, dy, acts=acts)

    # Return doubled batch
    # augmented_states = state_normalizer.to(device).normalize(augmented_states)
    new_obs = augmented_states # B, T, 7
    # new_acts = action_normalizer.to(device).normalize(augmented_acts)
    new_acts = augmented_acts
    if new_acts is not None:
        new_acts = new_acts.view(B, T, -1)
    return new_obs, new_acts

def rotate_and_displace(states, theta, dx, dy, acts=None, center=(250, 250)):
    """
    Input:
        states: (B, T, 8)
        acts: (B, T, 2) - optional
        theta: (B,) array of rotation angles in radians
        dx: (B,) array of displacements in x
        dy: (B,) array of displacements in y
        center: tuple of (x, y), rotation center

    Output:
        rotated and displaced states (B, T, 8)
    """

    B, T, D = states.shape
    assert D == 8

    cos_t = torch.cos(theta)  # (B,)
    sin_t = torch.sin(theta)  # (B,)

    R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t,  cos_t], dim=-1)
    ], dim=-2)  # (B, 2, 2)

    dot_xy = states[..., 0:2]     # (B, T, 2)
    T_xy = states[..., 2:4]       # (B, T, 2)
    T_sin = states[..., 4]        # (B, T)
    T_cos = states[..., 5]        # (B, T)
    dot_v = states[..., 6:8]      # (B, T, 2)

    # Step 1: Displace positions
    displacement = torch.stack([dx, dy], dim=-1).unsqueeze(1)  # (B, 1, 2)
    dot_xy_disp = dot_xy + displacement
    T_xy_disp = T_xy + displacement

    # Step 2: Rotate around new center
    center_tensor = torch.tensor(center, dtype=states.dtype, device=states.device).unsqueeze(0)  # (1, 2)
    new_center = center_tensor + displacement[:, 0, :]  # (B, 2)
    dot_xy_centered = dot_xy_disp - new_center.unsqueeze(1)  # (B, T, 2)
    T_xy_centered = T_xy_disp - new_center.unsqueeze(1)      # (B, T, 2)

    dot_xy_rot = torch.matmul(dot_xy_centered, R.transpose(1, 2)) + new_center.unsqueeze(1)
    T_xy_rot = torch.matmul(T_xy_centered, R.transpose(1, 2)) + new_center.unsqueeze(1)
    dot_v_rot = torch.matmul(dot_v, R.transpose(1, 2))

    # Step 3: Rotate target angle
    original_T_angle = torch.atan2(T_sin, T_cos)           # (B, T)
    T_angle_rot = original_T_angle + theta.unsqueeze(1)    # (B, T)
    T_angle_rot = (T_angle_rot + torch.pi) % (2 * torch.pi) - torch.pi

    T_sin_rot = torch.sin(T_angle_rot)
    T_cos_rot = torch.cos(T_angle_rot)

    out = torch.cat([
        dot_xy_rot,              
        T_xy_rot,                
        T_sin_rot.unsqueeze(-1), 
        T_cos_rot.unsqueeze(-1), 
        dot_v_rot                
    ], dim=-1)  # (B, T, 8)

    acts_rot = None
    if acts is not None:
        acts_rot = torch.matmul(acts, R.transpose(-1, -2))
    return out, acts_rot

def get_max_dx_dy(states):
    """
    Compute the maximum allowed positive/negative dx, dy displacements
    such that dot and T positions remain within the screen limits.

    Input:
        states: (B, T, 8), unnormalized
    Returns:
        dx_min: (B,), dx_max: (B,), dy_min: (B,), dy_max: (B,)
    """
    dot_x_min, dot_x_max = 50, 450
    dot_y_min, dot_y_max = 50, 450
    T_x_min, T_x_max = 100, 400
    T_y_min, T_y_max = 100, 400

    dot_x = states[:, :, 0]
    dot_y = states[:, :, 1]
    T_x = states[:, :, 2]
    T_y = states[:, :, 3]

    # For each position, compute min/max allowed shift before hitting boundary
    dot_dx_min = torch.where(dot_x < dot_x_min, torch.zeros_like(dot_x), dot_x_min - dot_x)
    dot_dx_max = torch.where(dot_x > dot_x_max, torch.zeros_like(dot_x), dot_x_max - dot_x)
    dot_dy_min = torch.where(dot_y < dot_y_min, torch.zeros_like(dot_y), dot_y_min - dot_y)
    dot_dy_max = torch.where(dot_y > dot_y_max, torch.zeros_like(dot_y), dot_y_max - dot_y)

    T_dx_min = torch.where(T_x < T_x_min, torch.zeros_like(T_x), T_x_min - T_x)
    T_dx_max = torch.where(T_x > T_x_max, torch.zeros_like(T_x), T_x_max - T_x)
    T_dy_min = torch.where(T_y < T_y_min, torch.zeros_like(T_y), T_y_min - T_y)
    T_dy_max = torch.where(T_y > T_y_max, torch.zeros_like(T_y), T_y_max - T_y)

    dx_min = torch.ceil(torch.maximum(dot_dx_min.max(dim=1).values, T_dx_min.max(dim=1).values)).to(torch.float32)
    dx_max = torch.floor(torch.minimum(dot_dx_max.min(dim=1).values, T_dx_max.min(dim=1).values)).to(torch.float32)
    dy_min = torch.ceil(torch.maximum(dot_dy_min.max(dim=1).values, T_dy_min.max(dim=1).values)).to(torch.float32)
    dy_max = torch.floor(torch.minimum(dot_dy_max.min(dim=1).values, T_dy_max.min(dim=1).values)).to(torch.float32)

    return dx_min, dx_max, dy_min, dy_max