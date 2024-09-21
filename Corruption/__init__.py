from corruption import *

corruption_list = [gaussian_noise, shot_noise, impulse_noise, defocus_blur,
                    glass_blur, motion_blur, zoom_blur, snow, frost, fog, contrast,spatter,
                    brightness,elastic_transform, pixelate,
                    jpeg_compression, speckle_noise, gaussian_blur,
                    saturate]

    # cv.imwrite(f"{corruption_func.__name__}.jpg",corrupted_image)



def corrupt_image(image,corruption_func,severity):
    corrupted_image=corruption_func(image,severity=5)
    print(corrupted_image)
    print("--"*80)
    corrupted_image=np.uint8(corrupted_image)
    print(corrupted_image)
    return corrupted_image 