#ifndef IMAGE_PROCESSING_VIT_H
#define IMAGE_PROCESSING_VIT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char **images;   // Array of image paths (strings)
    size_t size;     // Number of images in the list
} ImageList;

/**
 * This function will create a list of image paths from an array of image file paths.
 * It assumes the input is a list of strings, where each string is a valid image file path.
 *
 * @param input_images A pointer to an array of strings (image file paths).
 * @param num_images The number of image paths in the input array.
 * @return An ImageList structure containing the image paths.
 */
ImageList make_list_of_images(char **input_images, size_t num_images) {
    // Allocate memory for ImageList
    ImageList list;
    list.images = (char **)malloc(num_images * sizeof(char *));
    list.size = num_images;

    // Copy each image path into the list
    for (size_t i = 0; i < num_images; i++) {
        list.images[i] = (char *)malloc((strlen(input_images[i]) + 1) * sizeof(char));
        strcpy(list.images[i], input_images[i]);
    }

    return list;
}

/**
 * Function to free the memory allocated for the image list.
 * @param list The ImageList to free.
 */
void free_image_list(ImageList *list) {
    for (size_t i = 0; i < list->size; i++) {
        free(list->images[i]);
    }
    free(list->images);
}

#endif // IMAGE_PROCESSING_VIT_H
