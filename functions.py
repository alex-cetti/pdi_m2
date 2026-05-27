from main import *
### Pré processamento
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
    #blurred = cv2.GaussianBlur(equalized_img, (5, 5), 0)
    
    final_img = cv2.normalize(equalized_img, 
                             None, 
                             alpha=0, 
                             beta=255, 
                             norm_type=cv2.NORM_MINMAX, 
                             dtype=cv2.CV_32F
                )
    ##bb = final_img - blurred
    return final_img


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


## FILTRAGEM DE FREQUENCIA

## PASSA ALTA
def createPA(shape, center, radius, lpType, n):
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.power(c, 2.0) + np.power(r, 2.0)
    lpFilter_matrix = np.zeros(shape, np.float32)
    lpFilter = ""
    if lpType == "Ideal":  # Ideal high pass filter
        lpFilter = np.copy(d)
        lpFilter[lpFilter < pow(radius, 2.0)] = 0
        lpFilter[lpFilter >= pow(radius, 2.0)] = 1
    elif lpType == "Butterworth": #Butterworth Highpass Filters 
        lpFilter = 1.0 - 1.0 / (1 + np.power(np.sqrt(d)/radius, 2*n))
    elif lpType == "Gaussian": # Gaussian Highpass Filter 
        lpFilter = 1.0 - np.exp(-d/(2*pow(radius, 2.0)))
    else:
        raise ValueError(
            f"lpType inválido: {lpType!r}. "
            "Use: 'Ideal', 'Butterworth' ou 'Gaussian'."
        )
    


    lpFilter_matrix[:, :, 0] = lpFilter
    lpFilter_matrix[:, :, 1] = lpFilter

    return lpFilter_matrix

## PASSA BAIXA  
def createPB(shape, center, radius, lpType, n):
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.power(c, 2.0) + np.power(r, 2.0)
    lpFilter_matrix = np.zeros(shape, np.float32)
    lpFilter = ""
    if lpType == "Ideal":  # ideal low-pass filter
        lpFilter = np.copy(d)
        lpFilter[lpFilter < pow(radius, 2.0)] = 1
        lpFilter[lpFilter >= pow(radius, 2.0)] = 0
    elif lpType == "Butterworth": #Butterworth low-pass filter 
        lpFilter = 1.0 / (1 + np.power(np.sqrt(d)/radius, 2*n))
    elif lpType == "Gaussian": # Gaussian low pass filter
        lpFilter = np.exp(-d/(2*pow(radius, 2.0)))
    lpFilter_matrix[:, :, 0] = lpFilter
    lpFilter_matrix[:, :, 1] = lpFilter

    return lpFilter_matrix

## PASSA FAIXA
def createPF(shape, center, lpType, n,bandCenter, bandWidth ):
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.sqrt(np.power(c, 2.0) + np.power(r, 2.0))
    lpFilter_matrix = np.zeros(shape, np.float32)
    if lpType == "Ideal":  # Ideal bandpass filter
        lpFilter = np.copy(d)
        lpFilter[:, :] = 1
        lpFilter[d > (bandCenter+bandWidth/2)] = 0
        lpFilter[d < (bandCenter-bandWidth/2)] = 0
    elif lpType == "Butterworth": #Butterworth bandpass filter
        lpFilter = 1.0 - 1.0 / (1 + np.power(d*bandWidth/(d - pow(bandCenter,2)), 2*n))
    elif lpType == "Gaussian": # Gaussian bandpass filter
        lpFilter = np.exp(-pow((d-pow(bandCenter,2))/(d*bandWidth), 2))
    lpFilter_matrix[:, :, 0] = lpFilter
    lpFilter_matrix[:, :, 1] = lpFilter
    return lpFilter_matrix
## REJEITA FAIXA
def createRF(shape, center, lpType, n,bandCenter, bandWidth ):
    
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.sqrt(np.power(c, 2.0) + np.power(r, 2.0))
    lpFilter_matrix = np.zeros(shape, np.float32)
    if lpType == "Ideal":  # Ideal band stop filter
        lpFilter = np.copy(d)
        lpFilter[:, :] = 0
        lpFilter[d > (bandCenter+bandWidth/2)] = 1
        lpFilter[d < (bandCenter-bandWidth/2)] = 1
    elif lpType == "Butterworth": #Butterworth band stop filter
        lpFilter = 1.0 / (1 + np.power(d*bandWidth/(d - pow(bandCenter,2)), 2*n))
    elif lpType == "Gaussian": # Gaussian band stop filter
        lpFilter = 1 - np.exp(-pow((d-pow(bandCenter,2))/(d*bandWidth), 2))
    lpFilter_matrix[:, :, 0] = lpFilter
    lpFilter_matrix[:, :, 1] = lpFilter
    return lpFilter_matrix

def apply_filtro_frequencia(img, j_param):
    print(j_param['bandWidth'])
    dft = cv2.dft(img, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    ### 
    
    nrows, ncols = dft_shift.shape[:2] 
    real = np.power(dft_shift[:, :, 0], 2.0)
    imaginary = np.power(dft_shift[:, :, 1], 2.0)
    amplitude = np.sqrt(real+imaginary)
    minValue, maxValue, minLoc, maxLoc = cv2.minMaxLoc(amplitude)
    shap = dft_shift.shape
    center = (ncols // 2, nrows // 2)


    if j_param['filtro'] == "Passa Alta":
        mask = createPA(shap, 
                        center, 
                        j_param['radius'],
                        j_param['tipo'], 
                        j_param['n_param']
                    )
    elif j_param['filtro'] == "Passa Baixa":
        mask = createPB(shap, 
                        center, 
                        j_param['radius'],
                        j_param['tipo'], 
                        j_param['n_param']
                    )
    elif j_param['filtro'] == "Passa Faixa":
        mask = createPF(shap, 
                        center, 
                        j_param['tipo'],
                        j_param['n_param'], 
                        j_param['bandCenter'], 
                        j_param['bandWidth']
                    )
    elif j_param['filtro'] == "Rejeita Faixa":
        mask = createRF(shap, 
                        center, 
                        j_param['tipo'],
                        j_param['n_param'], 
                        j_param['bandCenter'], 
                        j_param['bandWidth']
                    )
    # mask2 = np.dstack([mask, mask])
    
    filtered_freq = dft_shift*mask
    f_ishift = np.fft.ifftshift(filtered_freq)  #inversa da fft
    img_back = cv2.idft(f_ishift)        #inversa da dft
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])  #recuperando a imagem capturando a magnitude (intesidade) 


    img_back = np.array(img_back, dtype=np.float32)
    filtered_img = np.abs(img_back)
    filtered_img -= filtered_img.min()
    filtered_img = filtered_img*255 / filtered_img.max()
    final_img = filtered_img.astype(np.uint8)

    return final_img



### MORFOLOGIA

#Kernel
def criar_kernel(formato, tamanho):

    """
    Cria elementos estruturantes para operações morfológicas.
    Formatos:
    - "quadrado"
    - "cruz"
    - "circulo"
    - "retangulo
    """
    if isinstance(tamanho, int):
        altura = largura = tamanho
    else:
        altura, largura = tamanho

    if altura % 2 == 0 or largura % 2 == 0:
        raise ValueError("Valor invalido pro kernel, use valores ímpars")

    if formato == "quadrado":
        kernel = np.ones((altura, largura), dtype=np.uint8)

    elif formato == "cruz":
        kernel = np.zeros((altura, largura), dtype=np.uint8)

        centro_y = altura // 2
        centro_x = largura // 2

        kernel[centro_y, :] = 1
        kernel[:, centro_x] = 1

    elif formato == "circulo":
        kernel = np.zeros((altura, largura), dtype=np.uint8)

        centro = tamanho // 2
        raio = centro

        for y in range(altura):
            for x in range(largura):
                distancia = np.sqrt((y - centro) ** 2 + (x - centro) ** 2)

                if distancia <= raio:
                    kernel[y, x] = 1

    elif formato == "retangulo":
        kernel = np.ones((altura, largura), dtype=np.uint8)

    return kernel


#Erosão
def erosao(img, kernel, objeto="preto"):
    
    """
    Aplica erosão em uma imagem limiarizada.

    objeto:
    - "branco": considera pixels brancos
    - "preto": considera pixels pretos
    """

    img = (img > 0).astype(np.uint8)
    k = (kernel > 0).astype(np.uint8)

    altura, largura = img.shape
    kh, kw = k.shape

    pad_y = kh // 2
    pad_x = kw // 2

    if objeto == "branco":
        valor_objeto = 1
        valor_fundo = 0

    elif objeto == "preto":
        valor_objeto = 0
        valor_fundo = 1

    #padding com valor de fundo
    img_padded = np.pad(
        img,
        ((pad_y, pad_y), (pad_x, pad_x)),
        mode="constant",
        constant_values=valor_fundo
    )

    #Resultado começa preenchido com fundo
    resultado = np.full_like(img, valor_fundo, dtype=np.uint8)

    #percorre a imagem
    for y in range(altura):
        for x in range(largura):

            img_igual_kernel = True

            #percorre o kernel
            for i_kernel in range(kh):
                for j_kernel in range(kw):

                    if k[i_kernel, j_kernel] == 1:

                        pixel = img_padded[y + i_kernel, x + j_kernel]

                        #confere se o pixel não bate com o kernel
                        if pixel != valor_objeto:
                            img_igual_kernel = False

            if img_igual_kernel == True:
                resultado[y, x] = valor_objeto
            else:
                resultado[y, x] = valor_fundo

    resultado = resultado * 255
    resultado = resultado.astype(np.uint8)

    return resultado


#Dilatação
def dilatacao(img, kernel, objeto="preto"):
    """
    Aplica dilatação em uma imagem limiarizada.

    objeto:
    - "branco": considera pixels brancos
    - "preto": considera pixels pretos
    """

    img = (img > 0).astype(np.uint8)
    k = (kernel > 0).astype(np.uint8)

    altura, largura = img.shape
    kh, kw = k.shape

    pad_y = kh // 2
    pad_x = kw // 2

    if objeto == "branco":
        valor_objeto = 1
        valor_fundo = 0

    elif objeto == "preto":
        valor_objeto = 0
        valor_fundo = 1

    #padding com valor de fundo
    img_padded = np.pad(
        img,
        ((pad_y, pad_y), (pad_x, pad_x)),
        mode="constant",
        constant_values=valor_fundo
    )

    #Resultado começa preenchido com fundo
    resultado = np.full_like(img, valor_fundo, dtype=np.uint8)

    #percorre a imagem
    for y in range(altura):
        for x in range(largura):

            encontrou_objeto = False

            #percorre o kernel
            for i_kernel in range(kh):
                for j_kernel in range(kw):

                    if k[i_kernel, j_kernel] == 1:

                        pixel = img_padded[y + i_kernel, x + j_kernel]

                        #confere se o pixel não bate com o kernel
                        if pixel == valor_objeto:
                            encontrou_objeto = True

            if encontrou_objeto == True:
                resultado[y, x] = valor_objeto
            else:
                resultado[y, x] = valor_fundo

    resultado = resultado * 255
    resultado = resultado.astype(np.uint8)

    return resultado



def apply_limiar(img_in, limiar):
    
    img_in = img_in.copy()
    img_out = np.zeros(img_in.shape)

    for i in range(img_in.shape[0]):
        for j in range(img_in.shape[1]):
            if img_in[i, j] > limiar:
                img_out[i, j] = 255
            else:
                img_out[i, j] = 0

    
    img_out = img_out.astype(np.uint8)

    return img_out



def abertura(img, kernel, objeto="branco"):
    """
    Aplica abertura em uma imagem limiarizada.

    Abertura = erosão + dilatação.
    """
    img_erodida = erosao(img, kernel, objeto)
    img_aberta = dilatacao(img_erodida, kernel, objeto)

    return img_aberta


def fechamento(img, kernel, objeto="branco"):
    """
    Aplica fechamento em uma imagem limiarizada.

    Fechamento = dilatação + erosão.
    """
    img_dilatada = dilatacao(img, kernel, objeto)
    img_fechada = erosao(img_dilatada, kernel, objeto)

    return img_fechada