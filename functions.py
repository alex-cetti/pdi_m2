from main import *

def gabarito(img_path, gab_path):

    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    
    tree = ET.parse(gab_path)
    for elem in tree.iter():
        if 'object' in elem.tag or 'part' in elem.tag:
            for attr in list(elem):
                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            xmin = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            ymin = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            xmax = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            ymax = int(round(float(dim.text)))
                
                    
                    cv2.rectangle(image, (xmin, ymin),
                                (xmax, ymax), (0, 255, 0), 2)
    
    return image


def normalizar_raw_image(img_raw):
    x0, x1 = 500, 1600
    y0, y1 = 50, 1080
    
    croped = img_raw[y0:y1, x0:x1]

    resized = cv2.resize(croped, (320, 320), interpolation=cv2.INTER_AREA)
    gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    equalized_img = cv2.equalizeHist(gray_img)

    final_img = cv2.normalize(equalized_img, 
                             None, 
                             alpha=0, 
                             beta=255, 
                             norm_type=cv2.NORM_MINMAX, 
                             dtype=cv2.CV_32F
                )

    return final_img

## PASSA BAIXA
def createPB(shape, center, radius, lpType=2, n=2):
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.power(c, 2.0) + np.power(r, 2.0)
    lpFilter_matrix = np.zeros(shape, np.float32)
    if lpType == 0:  # ideal low-pass filter
        lpFilter = np.copy(d)
        lpFilter[lpFilter < pow(radius, 2.0)] = 1
        lpFilter[lpFilter >= pow(radius, 2.0)] = 0
    elif lpType == 1: #Butterworth low-pass filter 
        lpFilter = 1.0 / (1 + np.power(np.sqrt(d)/radius, 2*n))
    elif lpType == 2: # Gaussian low pass filter
        lpFilter = np.exp(-d/(2*pow(radius, 2.0)))
    lpFilter_matrix[:, :, 0] = lpFilter
    lpFilter_matrix[:, :, 1] = lpFilter
    return lpFilter_matrix
## PASSA ALTA

def apply_PA(img, tipo, radius, n_param):
    dft = cv2.dft(img, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    ### Spectro de Magnuted, n entendi essa porra mas funciona
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])) 
    magnitude_spectrum = np.abs(magnitude_spectrum)
    magnitude_spectrum -= magnitude_spectrum.min()
    magnitude_spectrum = magnitude_spectrum*255 / magnitude_spectrum.max()
    magnitude_spectrum = magnitude_spectrum.astype(np.uint8)


    nrows, ncols = dft_shift.shape[:2] 
    real = np.power(dft_shift[:, :, 0], 2.0)
    imaginary = np.power(dft_shift[:, :, 1], 2.0)
    amplitude = np.sqrt(real+imaginary)
    minValue, maxValue, minLoc, maxLoc = cv2.minMaxLoc(amplitude)
    shap = dft_shift.shape
    center=maxLoc


    #maskPA = create_PA(dft_shift.shape, center=maxLoc, radius, tipo, n_param)
    maskPA = create_PA(shap, center, radius, tipo, n_param)
    filtered_freq = dft_shift*maskPA
    f_ishift = np.fft.ifftshift(filtered_freq)  #inversa da fft
    img_back = cv2.idft(f_ishift)        #inversa da dft
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])  #recuperando a imagem capturando a magnitude (intesidade) 
    

    img_back = np.array(img_back, dtype=np.float32)
    filtered_img = np.abs(img_back)
    filtered_img -= filtered_img.min()
    filtered_img = filtered_img*255 / filtered_img.max()
    final_img = filtered_img.astype(np.uint8)

    return final_img



def create_PA(shape, center, radius, lpType, n):
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.power(c, 2.0) + np.power(r, 2.0)
    lpFilter_matrix = np.zeros(shape, np.float32)
    if lpType == "Ideal":  # Ideal high pass filter
        lpFilter = np.copy(d)
        lpFilter[lpFilter < pow(radius, 2.0)] = 0
        lpFilter[lpFilter >= pow(radius, 2.0)] = 1
    elif lpType == "Butterworth": #Butterworth Highpass Filters 
        lpFilter = 1.0 - 1.0 / (1 + np.power(np.sqrt(d)/radius, 2*n))
    elif lpType == "Gaussian": # Gaussian Highpass Filter 
        lpFilter = 1.0 - np.exp(-d/(2*pow(radius, 2.0)))
    lpFilter_matrix[:, :, 0] = lpFilter
    lpFilter_matrix[:, :, 1] = lpFilter
    return lpFilter_matrix
## PASSA FAIXA
def createPF(shape, center, bandCenter, bandWidth, lpType=2, n=2):
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.sqrt(np.power(c, 2.0) + np.power(r, 2.0))
    lpFilter_matrix = np.zeros(shape, np.float32)
    if lpType == 0:  # Ideal bandpass filter
        lpFilter = np.copy(d)
        lpFilter[:, :] = 1
        lpFilter[d > (bandCenter+bandWidth/2)] = 0
        lpFilter[d < (bandCenter-bandWidth/2)] = 0
    elif lpType == 1: #Butterworth bandpass filter
        lpFilter = 1.0 - 1.0 / (1 + np.power(d*bandWidth/(d - pow(bandCenter,2)), 2*n))
    elif lpType == 2: # Gaussian bandpass filter
        lpFilter = np.exp(-pow((d-pow(bandCenter,2))/(d*bandWidth), 2))
    lpFilter_matrix[:, :, 0] = lpFilter
    lpFilter_matrix[:, :, 1] = lpFilter
    return lpFilter_matrix
## REJEITA FAIXA
def createRF(shape, center, bandCenter, bandWidth, lpType=2, n=2):
    
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.sqrt(np.power(c, 2.0) + np.power(r, 2.0))
    lpFilter_matrix = np.zeros(shape, np.float32)
    if lpType == 0:  # Ideal band stop filter
        lpFilter = np.copy(d)
        lpFilter[:, :] = 0
        lpFilter[d > (bandCenter+bandWidth/2)] = 1
        lpFilter[d < (bandCenter-bandWidth/2)] = 1
    elif lpType == 1: #Butterworth band stop filter
        lpFilter = 1.0 / (1 + np.power(d*bandWidth/(d - pow(bandCenter,2)), 2*n))
    elif lpType == 2: # Gaussian band stop filter
        lpFilter = 1 - np.exp(-pow((d-pow(bandCenter,2))/(d*bandWidth), 2))
    lpFilter_matrix[:, :, 0] = lpFilter
    lpFilter_matrix[:, :, 1] = lpFilter
    return lpFilter_matrix