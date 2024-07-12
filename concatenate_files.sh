#!/bin/bash

# Default directory and output file
DEFAULT_DIRECTORY="."
DEFAULT_OUTPUT_FILE="concatenated_code.txt"

# Directory containing the files to concatenate (defaults to current directory)
DIRECTORY=${1:-$DEFAULT_DIRECTORY}

# Output file (defaults to concatenated_code.txt)
OUTPUT_FILE=${2:-$DEFAULT_OUTPUT_FILE}

# Directories and files to exclude
# shellcheck disable=SC2054
EXCLUDE_DIRS_AND_FILES=(".venv" ".env" "FullStack" "tests" ".gitignore" "app_structure.json" "concatenate_files.sh" "concatenated_code.txt" "poetry.lock" "poetry.toml" "README.md" "dist" "__pycache__")

# Function to check if a directory or file should be excluded
should_exclude() {
  local item=$1
  for exclude in "${EXCLUDE_DIRS_AND_FILES[@]}"; do
    if [[ "$item" == *"$exclude"* ]]; then
      return 0
    fi
  done
  return 1
}

# Create or empty the output file
> "$OUTPUT_FILE"

# Function to process a directory
process_directory() {
  local current_dir=$1
  local indent=$2

  # Print the directory structure
  echo "${indent}${current_dir}/" >> "$OUTPUT_FILE"

  # Process each item in the directory
  for item in "$current_dir"/*; do
    if [ -d "$item" ]; then
      # Check if the directory should be excluded
      if ! should_exclude "$item"; then
        process_directory "$item" "  $indent"
      fi
    elif [ -f "$item" ]; then
      # Check if the file should be excluded
      if ! should_exclude "$item"; then
        # Print the filename and its content
        echo "  ${indent}File: ${item#$DIRECTORY/}" >> "$OUTPUT_FILE"
        echo "  ${indent}Content:" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        echo "Processing file: $item"  # Add this line to print the file being processed
        sed "s/^/  $indent/" "$item" >> "$OUTPUT_FILE" || echo "Error processing file: $item"  # Indent file content and handle errors
        echo "" >> "$OUTPUT_FILE"
        echo "  ${indent}====" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
      fi
    fi
  done
}

# Start processing from the root directory
process_directory "$DIRECTORY" ""

echo "All files have been concatenated into $OUTPUT_FILE."
