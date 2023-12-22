from traceback import print_exc
from os.path import basename, splitext
from PIL import Image
from typing import List, Callable, Tuple, Union
import matplotlib.pyplot as plt
from numpy import array, clip, pad, zeros, uint8
from numpy.random import normal
from PIL import Image

def get_standard_deviation(img: Union[ListImage, ListImageRaw], mean: FloatOrNone=None, variance: FloatOrNone=None) -> float:
    variance = variance if variance is not None else get_variance(img, mean)

    std_dev = variance ** 0.5

    return std_dev

def get_mean(img: Union[ListImage, ListImageRaw]) -> float:
    rows = len(img)
    cols = len(img[0])

    mean_val = sum([sum([img[i][j] for j in range(cols)]) for i in range(rows)]) / (rows * cols)

    return mean_val

def get_variance(img: Union[ListImage, ListImageRaw], mean: FloatOrNone=None) -> float:
    rows = len(img)
    cols = len(img[0])

    mean = mean if mean is not None else get_mean(img)

    variance_val = sum([sum([(img[i][j] - mean) ** 2 for j in range(cols)]) for i in range(rows)]) / (rows * cols)

    return variance_val


def get_bsnr(blurred_img_no_noise: ListImage, restored_img: ListImage, noise_variance: float) -> float:
    rows = len(blurred_img_no_noise)
    cols = len(blurred_img_no_noise[0])

    return 10 * log10(sum([sum([(blurred_img_no_noise[i][j] - restored_img[i][j]) ** 2 for j in range(cols)]) for i in range(rows)]) / (rows * cols * noise_variance))

def get_isnr(ideal_img: ListImage, degraded_img: ListImage, restored_img: ListImage) -> float:
    rows = len(ideal_img)
    cols = len(ideal_img[0])

    return 10 * log10(sum([sum([(ideal_img[i][j] - degraded_img[i][j]) ** 2 for j in range(cols)]) for i in range(rows)]) / sum([sum([(ideal_img[i][j] - restored_img[i][j]) ** 2 for j in range(cols)]) for i in range(rows)]))


def create_filter_kernel(weights: FilterKernel) -> FilterKernel:
    filter_kernel = np.array(weights)
    filter_kernel_sum = np.sum(filter_kernel)
    return (filter_kernel / filter_kernel_sum).tolist() if filter_kernel_sum != 0 else filter_kernel.tolist()

def get_mirrored_image_function(image: ListImage, filter_kernel_sizes: Sizes2D) -> ImageFunction:
    N, M = len(image), len(image[0])
    extension_sizes = tuple(size // 2 for size in filter_kernel_sizes)

    def mirrored_image_function(i: int, j: int) -> int:
        ii = max(0, min(N - 1, i + (0 if 0 <= i < N else -i)))
        jj = max(0, min(M - 1, j + (0 if 0 <= j < M else -j)))
        return image[ii][jj]

    return mirrored_image_function

def linear_spatial_filtering_raw(image: ListImage, filter_kernel: FilterKernel) -> ListImageRaw:
    return convolve2d(image, filter_kernel, mode='same', boundary='symm')

def linear_spatial_filtering(image: ListImage, filter_kernel: FilterKernel) -> ListImage:
    filtered_image_raw = linear_spatial_filtering_raw(image, filter_kernel)
    return convertToProperImage(filtered_image_raw)

def get_2d_discrete_fourier_transform(discrete_function: DiscreteFunctionMatrix, sizes: Union[Sizes2D, None] = None, centered: bool = False, normalized: bool = False) -> DiscreteFourierTransform:
    sizes = sizes or (len(discrete_function), len(discrete_function[0]))
    np_discrete_function = np.array(discrete_function, dtype=complex128)
    np_fft_result = np.fft.fft2(np_discrete_function, s=sizes)
    np_result = np.fft.fftshift(np_fft_result) if centered else np_fft_result
    if normalized:
        np_result = np_result / np.sqrt(np.prod(sizes))
    return np_result.tolist()

def get_2d_inverse_discrete_fourier_transform(fourier_transform: DiscreteFourierTransform, centered: bool = False, normalized: bool = False) -> ListImage:
    np_fourier_transform = np.array(fourier_transform, dtype=complex128)
    if centered:
        np_fourier_transform = np.fft.ifftshift(np_fourier_transform)
    np_inverse_fft_result = np.fft.ifft2(np_fourier_transform)
    if normalized:
        np_inverse_fft_result = np_inverse_fft_result * np.sqrt(np.prod(np.shape(fourier_transform)))
    return convertToProperImage(np.abs(np_inverse_fft_result))

def get_2d_discrete_fourier_transform_magnitude(fourier_transform: DiscreteFourierTransform) -> List[List[float]]:
    return np.abs(fourier_transform).tolist()

def plot_2d_discrete_fourier_transform(fourier_transform: DiscreteFourierTransform, centered: bool = True) -> None:
    if centered:
        fourier_transform = np.fft.fftshift(np.array(fourier_transform)).tolist()
    plot2DMatrix(get_2d_discrete_fourier_transform_magnitude(fourier_transform))

def add_gaussian_additive_noise(image: ListImage, std_dev_coef: float) -> ListImage:
    mean, std_deviation = get_mean(image), get_standard_deviation(image)
    noise = generateGaussianNoise(mean, std_dev_coef * std_deviation, np.shape(image))
    noisy_image = (np.array(image) + noise - mean).tolist()
    return convertToProperImage(noisy_image)

def get_gaussian_psf(window_sizes: Sizes2D, sigma: float = 1.0, centered: bool = False, normalization: Literal['sum', 'pi'] = 'sum') -> PSFKernerl:
    center_x, center_y = 0, 0
    if centered:
        center_x = (window_sizes[0] - 1) // 2
        center_y = (window_sizes[1] - 1) // 2
    sigma_square = sigma ** 2
    psf = [[np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma_square)) for y in range(window_sizes[1])] for x in range(window_sizes[0])]
    denominator = 2 * np.pi * sigma_square if normalization == 'pi' else np.sum(psf)
    normalized_psf = np.array(psf) / denominator
    return normalized_psf.tolist()

def blur_image(image: ListImage, psf: PSFKernerl) -> ListImage:
    image_fourier_transform = get_2d_discrete_fourier_transform(image)
    psf_fourier_transform = get_2d_discrete_fourier_transform(psf)
    blurred_image_fourier_transform = np.multiply(image_fourier_transform, psf_fourier_transform).tolist()
    return get_2d_inverse_discrete_fourier_transform(blurred_image_fourier_transform)

def get_noise_variance(blurred_image: ListImage) -> float:
    filter_kernel = create_filter_kernel([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    filtered_image = linear_spatial_filtering_raw(blurred_image, filter_kernel)
    noise_approximation = np.abs(np.array(blurred_image) - filtered_image)
    noise_variance = get_variance(noise_approximation)
    return noise_variance

def wiener_filtration(blurred_image: ListImage, psf: PSFKernerl, alpha: float) -> ListImage:
    noise_variance = get_noise_variance(blurred_image)
    blurred_variance = get_variance(blurred_image)
    K = blurred_image.size * noise_variance / blurred_variance
    blurred_image_fourier_transform = get_2d_discrete_fourier_transform(blurred_image)
    psf_fourier_transform = get_2d_discrete_fourier_transform(psf)
    restored_image_fourier_transform = np.divide(np.multiply(blurred_image_fourier_transform, psf_fourier_transform.conj()), np.abs(psf_fourier_transform) ** 2 + alpha * K).tolist()
    return get_2d_inverse_discrete_fourier_transform(restored_image_fourier_transform)


def _get_noise_variance(blurred_img: ListImage) -> float:
    rows = len(blurred_img)
    cols = len(blurred_img[0])

    kernel = create_filter_kernel([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    filtered_img = linear_spatial_filtering_raw(blurred_img, kernel)

    noise_approx = [[abs(blurred_img[i][j] - filtered_img[i][j]) for j in range(cols)] for i in range(rows)]

    noise_var = get_variance(noise_approx)
    
    return noise_var

def padMatrixWithZeros(matrix: List[List], new_sizes: Sizes2D) -> List[List]:
    original = array(matrix)

    pad_height = (new_sizes[0] - len(matrix)) // 2
    pad_width = (new_sizes[1] - len(matrix[0])) // 2

    extra_padding_height = (new_sizes[0] - len(matrix)) % 2
    extra_padding_width = (new_sizes[1] - len(matrix[0])) % 2

    new_array = pad(array=original, pad_width=((pad_height, pad_height + extra_padding_height), (pad_width, pad_width + extra_padding_width)), mode='constant', constant_values=0) # type: ignore

    return new_array.tolist()

def surroundMatrixWithZeros(matrix: List[List], new_sizes: Sizes2D) -> List[List]:
    original = array(matrix)
    new = zeros(new_sizes, dtype=original.dtype)
    new[:original.shape[0], :original.shape[1]] = original
    return new.tolist()

def generateGaussianNoise(mean: float=0.0, std_dev: float=1.0, size:Union[Sizes2D, None]=None) -> List:
    return normal(mean, std_dev, size).tolist()

def convertToProperImage(image: List[List[float]]) -> ListImage:
    return clip(a=image, a_min=MIN_INTENSITY, a_max=MAX_INTENSITY).astype(uint8).tolist() # type: ignore

def convertToListImage(image: Image.Image) -> ListImage:
    return [[image.getpixel((j, i)) for j in range(image.width)] for i in range(image.height)]

def convertToPillowImage(image: ListImage) -> Image.Image:
    return Image.fromarray(array(image, dtype=uint8), mode='L')

def saveImage(image: Image.Image, path: str) -> None:
    return image.save(path, mode='L')

def plot2DMatrix(matrix: List[List[float]]) -> None:
    plt.imshow(array(matrix), cmap='viridis', origin='lower')
    plt.colorbar(label='Magnitude')
    plt.title('2D DFT Magnitude Plot')
    plt.xlabel('Frequency (u)')
    plt.ylabel('Frequency (v)')
    plt.show()



def execute(image_path: str) -> None:
    computed_directory = f'./{COMPUTED_DIRECTORY_NAME}/'

    file_name_with_extension = basename(image_path)
    file_name, file_extension = splitext(file_name_with_extension)

    with Image.open(image_path) as im:
        image = convertToListImage(im)

        fourier_transform = get_2d_discrete_fourier_transform(image)

        inversed = get_2d_inverse_discrete_fourier_transform(fourier_transform)

        saveImage(convertToPillowImage(inversed), f'{computed_directory}{file_name}_inversed_original_{file_extension}')

        gaussian_psf = get_gaussian_psf((len(image), len(image[0])), sigma=5)

        blurred_image = blur_image(image, gaussian_psf)

        saveImage(convertToPillowImage(blurred_image), f'{computed_directory}{file_name}_blurred_{file_extension}')

        noisy_blurred_image = add_gaussian_additive_noise(blurred_image, 0.01)

        saveImage(convertToPillowImage(noisy_blurred_image), f'{computed_directory}{file_name}_noisy_blurred_{file_extension}')

        wiener_alphas = [0.00001, 0.0001, 0.001, 0.01]

        wiener_results = [wiener_filtration(noisy_blurred_image, gaussian_psf, alpha) for alpha in wiener_alphas]

        noise_variance = _get_noise_variance(noisy_blurred_image)

        bsnr_results = [get_bsnr(blurred_image, wiener_results[i], noise_variance) for i in range(len(wiener_results))]

        isnr_results = [get_isnr(image, noisy_blurred_image, wiener_results[i]) for i in range(len(wiener_results))]

        print(f'Image: {file_name_with_extension}')
        for i in range(len(wiener_alphas)):
            print(f'Alpha = {wiener_alphas[i]}, BSNR = {bsnr_results[i]}, ISNR = {isnr_results[i]}')
        print('\n')

        for i in range(len(wiener_results)):
            saveImage(convertToPillowImage(wiener_results[i]), f'{computed_directory}{file_name}_wiener{i + 1}_{file_extension}')


if __name__ == "__main__":
    try:
         execute('./assets/marcie.tif')
         execute('./assets/at3_1m4_02.tif')
    except Exception as e:
        print('Error occurred:')
        print_exc()