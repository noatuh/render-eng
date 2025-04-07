import pygame
import numpy as np
import math

# --- Configuration ---
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
BACKGROUND_COLOR = (0, 0, 0)  # Black
CUBE_COLORS = [
    (255, 255, 255),  # White
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255)     # Cyan
]
LINE_THICKNESS = 2

# Camera/Projection Settings
FOV = 90  # Field of view in degrees
NEAR_PLANE = 0.1
FAR_PLANE = 100.0
ASPECT_RATIO = SCREEN_WIDTH / SCREEN_HEIGHT

# --- 3D Object Definition (Cube) ---
# Vertices (x, y, z) centered around origin
cube_vertices = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
])

# Edges connecting vertex indices
cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
    (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
]

# --- 4D Object Definition (Tesseract) ---
# Generate all 16 vertices of a tesseract (4D hypercube)
tesseract_vertices = np.array([
    # First cube (w = -1)
    [-1, -1, -1, -1], [1, -1, -1, -1], [1, 1, -1, -1], [-1, 1, -1, -1],
    [-1, -1, 1, -1], [1, -1, 1, -1], [1, 1, 1, -1], [-1, 1, 1, -1],
    # Second cube (w = 1)
    [-1, -1, -1, 1], [1, -1, -1, 1], [1, 1, -1, 1], [-1, 1, -1, 1],
    [-1, -1, 1, 1], [1, -1, 1, 1], [1, 1, 1, 1], [-1, 1, 1, 1]
])

# Edges of a tesseract
tesseract_edges = [
    # Edges of the first cube (w = -1)
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
    
    # Edges of the second cube (w = 1)
    (8, 9), (9, 10), (10, 11), (11, 8),
    (12, 13), (13, 14), (14, 15), (15, 12),
    (8, 12), (9, 13), (10, 14), (11, 15),
    
    # Edges connecting the two cubes (connecting the 4th dimension)
    (0, 8), (1, 9), (2, 10), (3, 11),
    (4, 12), (5, 13), (6, 14), (7, 15)
]

# --- Transformation Matrices ---
def create_projection_matrix(fov_deg, aspect_ratio, near, far):
    """Creates a perspective projection matrix."""
    fov_rad = math.radians(fov_deg)
    f = 1.0 / math.tan(fov_rad / 2.0)
    matrix = np.zeros((4, 4))
    matrix[0, 0] = f / aspect_ratio
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix

def create_rotation_matrix_y(angle_deg):
    """Creates a rotation matrix around the Y axis."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    matrix = np.identity(4)
    matrix[0, 0] = cos_a
    matrix[0, 2] = sin_a
    matrix[2, 0] = -sin_a
    matrix[2, 2] = cos_a
    return matrix

def create_rotation_matrix_x(angle_deg):
    """Creates a rotation matrix around the X axis."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    matrix = np.identity(4)
    matrix[1, 1] = cos_a
    matrix[1, 2] = -sin_a
    matrix[2, 1] = sin_a
    matrix[2, 2] = cos_a
    return matrix

def create_translation_matrix(tx, ty, tz):
    """Creates a translation matrix."""
    matrix = np.identity(4)
    matrix[0, 3] = tx
    matrix[1, 3] = ty
    matrix[2, 3] = tz
    return matrix

# 4D Rotation matrices
def create_4d_rotation_xy(angle_deg):
    """Create a 4D rotation matrix in the XY plane."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    matrix = np.identity(5)  # 5x5 for 4D homogeneous coordinates
    matrix[0, 0] = cos_a
    matrix[0, 1] = -sin_a
    matrix[1, 0] = sin_a
    matrix[1, 1] = cos_a
    return matrix

def create_4d_rotation_xz(angle_deg):
    """Create a 4D rotation matrix in the XZ plane."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    matrix = np.identity(5)
    matrix[0, 0] = cos_a
    matrix[0, 2] = -sin_a
    matrix[2, 0] = sin_a
    matrix[2, 2] = cos_a
    return matrix

def create_4d_rotation_xw(angle_deg):
    """Create a 4D rotation matrix in the XW plane."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    matrix = np.identity(5)
    matrix[0, 0] = cos_a
    matrix[0, 3] = -sin_a
    matrix[3, 0] = sin_a
    matrix[3, 3] = cos_a
    return matrix

def create_4d_rotation_yz(angle_deg):
    """Create a 4D rotation matrix in the YZ plane."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    matrix = np.identity(5)
    matrix[1, 1] = cos_a
    matrix[1, 2] = -sin_a
    matrix[2, 1] = sin_a
    matrix[2, 2] = cos_a
    return matrix

def create_4d_rotation_yw(angle_deg):
    """Create a 4D rotation matrix in the YW plane."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    matrix = np.identity(5)
    matrix[1, 1] = cos_a
    matrix[1, 3] = -sin_a
    matrix[3, 1] = sin_a
    matrix[3, 3] = cos_a
    return matrix

def create_4d_rotation_zw(angle_deg):
    """Create a 4D rotation matrix in the ZW plane."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    matrix = np.identity(5)
    matrix[2, 2] = cos_a
    matrix[2, 3] = -sin_a
    matrix[3, 2] = sin_a
    matrix[3, 3] = cos_a
    return matrix

# --- Main Function ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Interactive Python 3D/4D Renderer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Interactive controls
    angle_x = 0
    angle_y = 0
    angle_xy = 0  # 4D rotation angles
    angle_zw = 0
    angle_xw = 0
    cam_x, cam_y, cam_z = 0, 0, -5
    current_color_idx = 0
    auto_rotate = True
    rotation_speed_x = 0.5
    rotation_speed_y = 0.8
    rotation_speed_4d = 0.3
    zoom_level = FOV
    
    # Display mode (0 for cube, 1 for tesseract)
    display_mode = 0

    projection_matrix = create_projection_matrix(zoom_level, ASPECT_RATIO, NEAR_PLANE, FAR_PLANE)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    # Cycle through colors
                    current_color_idx = (current_color_idx + 1) % len(CUBE_COLORS)
                elif event.key == pygame.K_SPACE:
                    # Toggle auto-rotation
                    auto_rotate = not auto_rotate
                elif event.key == pygame.K_r:
                    # Reset view
                    angle_x, angle_y = 0, 0
                    angle_xy, angle_zw, angle_xw = 0, 0, 0
                    cam_x, cam_y, cam_z = 0, 0, -5
                    zoom_level = FOV
                    projection_matrix = create_projection_matrix(zoom_level, ASPECT_RATIO, NEAR_PLANE, FAR_PLANE)
                elif event.key == pygame.K_t:
                    # Toggle between cube and tesseract
                    display_mode = (display_mode + 1) % 2

        # Handle continuous key presses
        keys = pygame.key.get_pressed()
        
        # Camera rotation controls
        if keys[pygame.K_UP]:
            angle_x -= 2
        if keys[pygame.K_DOWN]:
            angle_x += 2
        if keys[pygame.K_LEFT]:
            angle_y -= 2
        if keys[pygame.K_RIGHT]:
            angle_y += 2
            
        # Camera movement controls
        if keys[pygame.K_w]:
            cam_y += 0.1
        if keys[pygame.K_s]:
            cam_y -= 0.1
        if keys[pygame.K_a]:
            cam_x -= 0.1
        if keys[pygame.K_d]:
            cam_x += 0.1
            
        # Zoom controls
        if keys[pygame.K_z]:  # Zoom in
            cam_z += 0.1
        if keys[pygame.K_x]:  # Zoom out
            cam_z -= 0.1

        # --- Update ---
        if auto_rotate:
            angle_x += rotation_speed_x
            angle_y += rotation_speed_y
            angle_xy += rotation_speed_4d
            angle_zw += rotation_speed_4d
            angle_xw += rotation_speed_4d * 0.7

        # --- Transformations ---
        rotation_x = create_rotation_matrix_x(angle_x)
        rotation_y = create_rotation_matrix_y(angle_y)
        translation = create_translation_matrix(cam_x, cam_y, cam_z)

        # Combine 3D transformations
        world_matrix = translation @ rotation_y @ rotation_x

        # --- Render ---
        screen.fill(BACKGROUND_COLOR)
        projected_points = []

        if display_mode == 0:  # Render cube
            for vertex in cube_vertices:
                # Convert vertex to homogeneous coordinates (add 1)
                point_h = np.append(vertex, 1.0)

                # Apply world transformation
                transformed_point = world_matrix @ point_h

                # Apply projection
                projected_point_h = projection_matrix @ transformed_point

                # Perspective divide (convert back from homogeneous to Cartesian)
                if projected_point_h[3] != 0 and projected_point_h[3] > 0:
                    w = projected_point_h[3]
                    projected_x = projected_point_h[0] / w
                    projected_y = projected_point_h[1] / w

                    # Convert NDC to screen coordinates
                    screen_x = int((projected_x + 1) * 0.5 * SCREEN_WIDTH)
                    screen_y = int((1 - (projected_y + 1) * 0.5) * SCREEN_HEIGHT)
                    projected_points.append((screen_x, screen_y))
                else:
                    projected_points.append(None)

            # Draw edges
            for edge in cube_edges:
                start_idx, end_idx = edge
                p1 = projected_points[start_idx]
                p2 = projected_points[end_idx]

                # Only draw if both points are valid (visible after projection)
                if p1 is not None and p2 is not None:
                    pygame.draw.line(screen, CUBE_COLORS[current_color_idx], p1, p2, LINE_THICKNESS)
                    
        else:  # Render tesseract
            # Create 4D rotation matrices (rotating in various 4D planes)
            rot_xy = create_4d_rotation_xy(angle_xy)
            rot_zw = create_4d_rotation_zw(angle_zw)
            rot_xw = create_4d_rotation_xw(angle_xw)
            
            # Project from 4D to 3D
            for vertex in tesseract_vertices:
                # Convert to 5D homogeneous coordinates
                point_h = np.append(vertex, 1.0)
                
                # Apply 4D rotations
                point_h_5d = np.zeros(5)
                point_h_5d[:4] = point_h[:4]
                point_h_5d[4] = 1.0
                
                # Apply various 4D rotations
                rotated_point = rot_xw @ rot_zw @ rot_xy @ point_h_5d
                
                # Project from 4D to 3D (simple perspective division by w component)
                scale_factor = 1.5 / (3 + rotated_point[3])
                point_3d = np.zeros(4)
                point_3d[0] = rotated_point[0] * scale_factor
                point_3d[1] = rotated_point[1] * scale_factor
                point_3d[2] = rotated_point[2] * scale_factor
                point_3d[3] = 1.0
                
                # Apply regular 3D transformations
                transformed_point = world_matrix @ point_3d
                
                # Apply 3D projection
                projected_point_h = projection_matrix @ transformed_point
                
                # Perspective divide as usual
                if projected_point_h[3] != 0 and projected_point_h[3] > 0:
                    w = projected_point_h[3]
                    projected_x = projected_point_h[0] / w
                    projected_y = projected_point_h[1] / w

                    # Convert NDC to screen coordinates
                    screen_x = int((projected_x + 1) * 0.5 * SCREEN_WIDTH)
                    screen_y = int((1 - (projected_y + 1) * 0.5) * SCREEN_HEIGHT)
                    projected_points.append((screen_x, screen_y))
                else:
                    projected_points.append(None)
                    
            # Draw tesseract edges
            for edge in tesseract_edges:
                start_idx, end_idx = edge
                if start_idx < len(projected_points) and end_idx < len(projected_points):
                    p1 = projected_points[start_idx]
                    p2 = projected_points[end_idx]

                    # Only draw if both points are valid (visible after projection)
                    if p1 is not None and p2 is not None:
                        pygame.draw.line(screen, CUBE_COLORS[current_color_idx], p1, p2, LINE_THICKNESS)
        
        # Display controls
        controls_text = [
            "Controls:",
            "Arrow keys: Rotate object",
            "WASD: Move camera",
            "Z/X: Zoom in/out",
            "C: Change color",
            "T: Toggle Cube/Tesseract",
            "Space: Toggle auto-rotation",
            "R: Reset view",
            f"Current mode: {'Cube' if display_mode == 0 else 'Tesseract'}"
        ]
        
        y_offset = 10
        for text in controls_text:
            text_surface = font.render(text, True, (200, 200, 200))
            screen.blit(text_surface, (10, y_offset))
            y_offset += 25

        pygame.display.flip()
        clock.tick(60)  # Limit FPS

    pygame.quit()

if __name__ == '__main__':
    main()