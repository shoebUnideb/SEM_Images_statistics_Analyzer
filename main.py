import cv2
import numpy as np
import os
from pathlib import Path
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
from skimage.segmentation import clear_border

def main():
    """
    Main function to process multiple images for grain analysis.
    Creates necessary directories and processes all images in input folder.
    """
    # Define input and output directories
    input_dir = Path("input_images")
    output_dir = Path("output_measurements")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different types of output
    csv_dir = output_dir / "csv_files"
    image_dir = output_dir / "processed_images"
    plot_dir = output_dir / "distribution_plots"
    
    # Create all directories
    csv_dir.mkdir(exist_ok=True)
    image_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    
    # Process all images in input directory
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    
    for image_path in input_dir.iterdir():
        if image_path.suffix.lower() in valid_extensions:
            try:
                process_single_image(image_path, csv_dir, image_dir, plot_dir)
                print(f"Successfully processed {image_path.name}")
            except Exception as e:
                print(f"Error processing {image_path.name}: {str(e)}")

def create_size_distribution_plots(areas, perimeters, output_path, image_name):
    """
    Create and save histograms for grain size distribution.
    
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot area distribution
    ax1.hist(areas, bins='auto', color='blue', alpha=0.7)
    ax1.set_title(f'Grain Area Distribution - {image_name}')
    ax1.set_xlabel('Area (µm²)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Plot perimeter distribution
    ax2.hist(perimeters, bins='auto', color='green', alpha=0.7)
    ax2.set_title(f'Grain Perimeter Distribution - {image_name}')
    ax2.set_xlabel('Perimeter (µm)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path / f"{Path(image_name).stem}_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

def process_single_image(image_path, csv_dir, image_dir, plot_dir):
    """
    Process a single image and save its results.
    
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Could not read the image file")
    
    height, _, _ = image.shape
    cropped_img = image[:height - 80, :, :]

    # Process image
    gray, thresh = preprocess_image(cropped_img)
    markers, segmented_image = perform_watershed_segmentation(cropped_img, thresh)
    
    # Calculate and save grain properties
    csv_filename = csv_dir / f"{image_path.stem}_measurements.csv"
    stats = calculate_grain_properties(markers, gray, csv_filename, plot_dir, image_path.name)
    
    # Print statistics
    print(f"\nResults for {image_path.name}:")
    print(f"Total number of particles: {stats['total_particles']}")
    print(f"Surface Coverage: {stats['surface_coverage']:.2f}%")
    print(f"Total grain area: {stats['total_grain_area']:.2f} µm²")
    print(f"Mean grain area: {stats['mean_grain_area']:.2f} µm²")
    print(f"Particle density: {stats['particle_density']:.6f} particles/µm²\n")
    
    # Save processed images
    colored_grains = color.label2rgb(markers, bg_label=0)
    colored_grains = (colored_grains * 255).astype(np.uint8)
    
    cv2.imwrite(str(image_dir / f"{image_path.stem}_segmented.jpg"), segmented_image)
    cv2.imwrite(str(image_dir / f"{image_path.stem}_colored.jpg"), cv2.cvtColor(colored_grains, cv2.COLOR_RGB2BGR))

def preprocess_image(image):
    """
    Preprocess the input image for grain analysis.
    
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return gray, thresh

def perform_watershed_segmentation(image, thresh):
    """
    Perform watershed segmentation on the image.
    """
    # Define kernel for morphological operations
    kernel = np.ones((3,3), np.uint8)
    
    # Perform morphological operations
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = clear_border(opening)  # Remove edge touching grains
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 10
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(image, markers)
    
    # Mark boundaries in yellow
    image_copy = image.copy()
    image_copy[markers == -1] = [0, 255, 255]
    
    return markers, image_copy

def calculate_grain_properties(markers, intensity_image, output_path, plot_dir, image_name, pixels_to_um=0.5):
    """
    Calculate and save grain properties to CSV file, including additional statistics.
    
    Args:
        markers: Image markers from watershed
        intensity_image: Original grayscale image
        output_path: Path to save the CSV file
        plot_dir: Directory to save distribution plots
        image_name: Name of the original image
        pixels_to_um: Conversion factor from pixels to micrometers
    
    Returns:
        dict: Dictionary containing calculated statistics
    """
    # Properties to measure
    prop_list = [
        'Area',
        'equivalent_diameter',
        'orientation',
        'MajorAxisLength',
        'MinorAxisLength',
        'Perimeter'
    ]
    
    # Get region properties
    regions = measure.regionprops(markers, intensity_image=intensity_image)
    
    # Sort regions by area to identify the background (typically largest)
    sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)
    
    # Remove the background (largest) region
    grain_regions = sorted_regions[1:]
    
    # Collect areas and perimeters for plotting
    areas = [region.area * (pixels_to_um ** 2) for region in grain_regions]
    perimeters = [region.perimeter * pixels_to_um for region in grain_regions]
    
    # Create and save distribution plots
    create_size_distribution_plots(areas, perimeters, plot_dir, image_name)
    
    # Calculate basic statistics
    total_particles = len(grain_regions)
    total_image_area_pixels = intensity_image.shape[0] * intensity_image.shape[1]
    total_image_area_um2 = total_image_area_pixels * (pixels_to_um ** 2)
    
    # Calculate total grain area in µm²
    total_grain_area_pixels = sum(region.area for region in grain_regions)
    total_grain_area_um2 = total_grain_area_pixels * (pixels_to_um ** 2)
    
    # Calculate mean grain area in µm²
    mean_grain_area_um2 = total_grain_area_um2 / total_particles if total_particles > 0 else 0
    
    # Calculate surface coverage percentage
    surface_coverage = (total_grain_area_pixels / total_image_area_pixels) * 100
    
    # Calculate particle density (particles per µm²)
    particle_density = total_particles / total_image_area_um2 if total_image_area_um2 > 0 else 0
    
    # Save measurements to CSV
    with open(output_path, 'w') as output_file:
        # Write global statistics first
        output_file.write("Global Statistics\n")
        output_file.write(f"Total number of particles:,{total_particles}\n")
        output_file.write(f"Surface Coverage (%):,{surface_coverage:.2f}\n")
        output_file.write(f"Total grain area (µm²):,{total_grain_area_um2:.2f}\n")
        output_file.write(f"Mean grain area (µm²):,{mean_grain_area_um2:.2f}\n")
        output_file.write(f"Particle density (particles/µm²):,{particle_density:.6f}\n")
        
        # Add distribution statistics
        output_file.write("\nSize Distribution Statistics\n")
        output_file.write(f"Mean area (µm²):,{np.mean(areas):.2f}\n")
        output_file.write(f"Median area (µm²):,{np.median(areas):.2f}\n")
        output_file.write(f"Standard deviation of area (µm²):,{np.std(areas):.2f}\n")
        output_file.write(f"Mean perimeter (µm):,{np.mean(perimeters):.2f}\n")
        output_file.write(f"Median perimeter (µm):,{np.median(perimeters):.2f}\n")
        output_file.write(f"Standard deviation of perimeter (µm):,{np.std(perimeters):.2f}\n")
        
        output_file.write("\nIndividual Grain Measurements\n")
        
        # Write header for individual measurements
        output_file.write('Grain #,' + ','.join(prop_list) + '\n')
        
        # Write measurements for each grain
        for idx, region_props in enumerate(grain_regions, 1):
            measurements = []
            measurements.append(str(idx))
            
            for prop in prop_list:
                value = region_props[prop]
                
                # Convert measurements based on property type
                if prop == 'Area':
                    value = value * (pixels_to_um ** 2)
                elif prop == 'orientation':
                    value = value * 57.2958  # Convert radians to degrees
                elif prop == 'Perimeter':
                    value = value * pixels_to_um
                
                # Round to 2 decimal places
                value = round(value, 2)
                measurements.append(str(value))
            
            output_file.write(','.join(measurements) + '\n')
    
    # Return statistics dictionary
    return {
        'total_particles': total_particles,
        'surface_coverage': surface_coverage,
        'total_grain_area': total_grain_area_um2,
        'mean_grain_area': mean_grain_area_um2,
        'particle_density': particle_density
    }

if __name__ == "__main__":
    main()