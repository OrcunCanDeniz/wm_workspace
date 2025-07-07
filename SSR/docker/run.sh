# Configure X11 forwarding for GUI visualization (e.g., Open3D)
# Docker image and container configuration
# Define the user's home directory using the standard HOME variable
HOME_DIR="$HOME"

# Define the project name
PROJECT_NAME="mmdet3d_plugin"

# List of projects to mount into the Docker container
PROJECTS_LIST=("SSR")

# Function to generate Docker volume mount strings for a project
# Arguments:
#   $1 - Project name
# Outputs:
#   A string of volume mount options for Docker
populate_volumes() {
    local project_name="$1"
    local data_path="$HOME_DIR/wm_workspace/$project_name/data"
    local code_path="$HOME_DIR/wm_workspace/$project_name"
    local docker_code_path="/workspace/$project_name"
    local volumes=""

    # Check if the data folder exists
    if [ ! -d "$data_path" ]; then
        # If no data folder, mount only the project directory
        volumes="--volume=$code_path:$docker_code_path"
    else
        # Process symbolic links in the data folder if it exists
        local all_softlinks
        all_softlinks=$(find "$data_path" -type l)
        for softlink in $all_softlinks; do
            local original_file
            local original_file_path
            local softlink_in_docker
            original_file=$(readlink "$softlink")
            original_file_path=$(realpath "$original_file")
            softlink_in_docker="$docker_code_path/data/$(basename "$softlink")"
            volumes="$volumes --volume=$original_file_path:$softlink_in_docker"
        done
        # Always mount the project directory as well
        volumes="$volumes --volume=$code_path:$docker_code_path"
    fi

    echo "$volumes"
}

TAG="latest"
IMAGE_NAME="${PROJECT_NAME,,}:${TAG,,}"
CONTAINER_NAME="${PROJECT_NAME,,}"


# Initialize volumes string
VOLUMES=""

# Populate volumes for each project in the list
for project in "${PROJECTS_LIST[@]}"; do
    volumes=$(populate_volumes "$project")
    VOLUMES="$VOLUMES $volumes"
done

# Remove leading/trailing spaces and collapse multiple spaces into one
VOLUMES_CLEAN=$(echo "$VOLUMES" | tr -s ' ' | sed 's/^ //;s/ $//')

# Display the volumes
echo "Mounting the following directories to the Docker container:"
echo "$VOLUMES_CLEAN" | tr ' ' '\n'


VISUAL="--env=DISPLAY \
        --env=QT_X11_NO_MITSHM=1 \
        --volume=/tmp/.X11-unix:/tmp/.X11-unix"
xhost +local:docker  # Allow Docker to access the X server

# Launch the Docker container
docker run -d -it --rm \
    -p 8888:8888 \
    $VOLUMES \
    $VISUAL \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    --gpus all \
    --privileged \
    --net=host \
    --ipc=host \
    --shm-size=30G \
    --name "$CONTAINER_NAME" \
    "$IMAGE_NAME"