clear
source = "Audio"

 

if (source == "Random bits")
    d = randi(15, 1, 128)
    data = dec2bin(d)-'0'
end
if (source == "Text file")
    [file, path] = uigetfile("*.txt"); 
    fileID = fopen(strcat(path, "/", file));
    text = fscanf(fileID, '%c')
    data = dec2bin(text, 8)-'0'
    fclose(fileID);
end
if (source == "Image")
    [file, ~] = uigetfile("*"); 
    im = imread(file);
    im = rgb2gray(im);
    im = imresize(im, [64, 64]);
    data = dec2bin(im)-'0'
end
if (source == "Audio")
    recordedObject = audiorecorder;
    disp("Recording started....")
    recordblocking(recordedObject, 10);
    Fs = recordedObject.SampleRate;
    disp('End of Recording.');
    y = getaudiodata(recordedObject)
    sign_of_y = sign(y);
    y = abs(y);
    data = fix(rem(y*pow2(-(8-1):8),2))
end

dim_before_coding = size(data);

global coding;
coding = "(7, 4) hamming code"

 

if (coding == "(7, 4) hamming code")
    bits = reshape(data, [], 4);
    coded_data = encode(bits, 7, 4)
    disp("Encoded with (7, 4) hamming code")
elseif (coding == "BCH (127, 64) code")
    bits = reshape(data, [], 64);
    bits = gf(bits);
    coded_data = bchenc(bits, 127, 64)
    disp("Encoded with BCH (127, 64) code")
    
else
    bits = reshape(data, [], 51);
    bits = gf(bits);
    coded_data = bchenc(bits, 63, 51)
    disp("Encoded with BCH (63, 51) code")
end
dim_after_coding = size(coded_data);

modulation = "32 QAM"

 

if (modulation == "16 FSK")
    modulating_signal = reshape(coded_data, 4, []);
    modulating_signal = binaryToDecimal(modulating_signal);
    modulated_signal = fskmod(modulating_signal, 16, 0.05, 4)
    disp("16 FSK modulation applied")
end
if (modulation == "64 QAM")
    modulating_signal = reshape(coded_data, 4, []);
    modulating_signal = binaryToDecimal(modulating_signal);
    modulated_signal = qammod(modulating_signal, 64)
    disp("64 QAM modulation applied")
end
if (modulation == "16 QAM")
    modulating_signal = reshape(coded_data, 4, []);
    modulating_signal = binaryToDecimal(modulating_signal);
    modulated_signal = qammod(modulating_signal, 16)
    disp("16 QAM modulation applied")
end
if (modulation == "32 QAM")
    modulating_signal = reshape(coded_data, 4, []);
    modulating_signal = binaryToDecimal(modulating_signal);
    modulated_signal = qammod(modulating_signal, 32)
    disp("32 QAM modulation applied")
end

global dim_after_modulation;
dim_after_modulation = size(modulated_signal);

 

if (modulation == "16 PSK" || modulation == "16 QAM")
    scatterplot(modulated_signal)
    title("Constellation diagram for ", modulation)
end

if (source == "Text file")
    disp(text)
end
if (source == "Image")
    imshow(im)
end
if (source == "Audio")
    figure;
    subplot(2, 1, 1)
    plot(y)
    xlabel("t")
    ylabel("x(t)")
    title("Audio Signal")
    
    subplot(2, 1, 2)
    x = y;
    N = length(x);
    xdft = fft(x);
    xdft = xdft(1:N/2+1);
    psdx = (1/(Fs*N)) * abs(xdft).^2;
    psdx(2:end-1) = 2*psdx(2:end-1);
    freq = 0:Fs/length(x):Fs/2;

    plot(freq,10*log10(psdx))
    grid on
    title('Periodogram Using FFT')
    xlabel('Frequency (Hz)')
    ylabel('Power/Frequency (dB/Hz)')
end

SNR = 28

 

transmitted_signal = awgn(modulated_signal, SNR, 'measured')

 

if (modulation == "16 PSK" || modulation == "16 QAM")
    scatterplot(transmitted_signal)
    title("Constellation diagram for ", modulation)
end
 

if (modulation == "16 FSK")
    demodulated_signal = fskdemod(transmitted_signal, 16, 0.05, 4)
    disp("Demodulation - 16 FSK")
end
if (modulation == "16 PSK")
    demodulated_signal = pskdemod(modulated_signal, 16)
    disp("16 PSK demodulation")
end
if (modulation == "16 QAM")
    demodulated_signal = qamdemod(modulated_signal, 16)
    disp("16 QAM demodulation")
end
if (modulation == "32 QAM")
    demodulated_signal = qamdemod(modulated_signal, 32)
    disp("32 QAM demodulation")
end

dim_after_demodulation = size(demodulated_signal);

demodulated_signal2 = changeShapeAfterDemodulation(demodulated_signal);

if (coding == "(7, 4) hamming code")
    temp = reshape(demodulated_signal2, dim_after_coding(1), dim_after_coding(2));
    decoded_signal = decode(temp, 7, 4, "hamming/binary")
    disp("Decoded using (7, 4) hamming code")
    
end
if (coding == "BCH (127, 64) code")
    temp = reshape(demodulated_signal2, dim_after_coding(1), dim_after_coding(2));
    temp = gf(temp);
    decoded_signal = bchdec(temp, 127, 64);
    decoded_signal = decoded_signal.x
    disp("Decoded using BCH (127, 64) code")
end

decoded_signal2 =reshape(decoded_signal, dim_before_coding(1), dim_before_coding(2))
 
ber_with_coding = sum(decoded_signal2 ~= data, "all") / (numel(data))


 

if (source == "Random bits")
    reconstructed_signal = bi2de(decoded_signal2, 'left-msb')
end
if (source == "Text file")
    text = binaryToText(decoded_signal2);
    text = char(text);
    reconstructed_signal2 = convertCharsToStrings(text)
end
if (source == "Image")
    decimals = binaryToText(decoded_signal2);
    reconstructed_signal2 = reshape(decimals, 64, 64);
    recovered_image = mat2gray(reconstructed_signal2);
    imshow(recovered_image)
end
if (source == "Audio")
    decoded_signal2 = decoded_signal2 * pow2(8-1:-1:-8).'
    sound(decoded_signal2, 8000, 8)
end

function text = binaryToText(inp)
    dim = size(inp);
    dim = dim(1);
    text = zeros(dim, 1);
    for i = 1:dim
        sum = 0;
        for j = 1:8
            sum = sum + inp(i, j) * 2^(8 - j);
        end
        text(i) = sum;
    end
end

function y = changeShapeAfterDemodulation(inp)
    global dim_after_modulation;
    dim = [4, dim_after_modulation(2)];
    y = zeros(dim);
    for i = 1:dim_after_modulation(2)
        x = dec2bin(inp(i), 4);
        for j = 1:4
            y(5-j, i) = str2double(x(j));
        end
    end
end

function arr = binaryToDecimal(inp)
    if (class(inp) == "gf")
        inp = inp.x;
    end
    dim = size(inp);
    arr = zeros(1,dim(2));
    
    for i = 1:dim(2)
        vect = inp(:, i);
        for j = 1:length(vect)
            arr(i) = arr(i) + vect(j)*2^(j-1);
        end
    end
end