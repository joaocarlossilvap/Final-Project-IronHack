import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.mention import mention

with st.sidebar: 
    st.image('https://seeklogo.com/images/I/ironhack-logo-F751CF4738-seeklogo.com.png')
    st.title('LipNet Project')

colored_header(
    label="Building a ML Model",
    description="Where the magic happens",
    color_name="violet-70",
)

code = '''def load_video(path:str) -> List[float]: 

    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std'''
st.code(code, language='python')

with st.expander("Loads the Video and Normalize the Data"):
    st.write("1. `def load_video(path:str) -> List[float]:`")
    st.write("This line defines a Python function called `load_video`.")
    st.write("The function takes a single argument `path`, which is expected to be a string representing the file path of a video.")
    st.write("The function returns a list of floating-point numbers.")

    st.write("2. `cap = cv2.VideoCapture(path)`")
    st.write("This line creates a video capture object `cap` using the OpenCV library to read the video located at the given `path`.")

    st.write("3. `frames = []`")
    st.write("This line initializes an empty list called `frames` to store the frames of the video after processing.")

    st.write("4. `for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):`")
    st.write("This line starts a loop that iterates over each frame of the video.")
    st.write("It runs for the number of frames in the video, which is obtained using `cap.get(cv2.CAP_PROP_FRAME_COUNT)`.")

    st.write("5. `ret, frame = cap.read()`")
    st.write("This line reads the next frame of the video.")
    st.write("The variable `ret` is a boolean that indicates whether the frame was successfully read, and `frame` stores the actual frame data.")

    st.write("6. `frame = tf.image.rgb_to_grayscale(frame)`")
    st.write("This line converts the color frame to grayscale using TensorFlow's `tf.image.rgb_to_grayscale()` function.")

    st.write("7. `frames.append(frame[190:236,80:220,:])`")
    st.write("This line extracts a specific region of interest (ROI) from the frame using array slicing.")
    st.write("It selects rows 190 to 235 and columns 80 to 219 of the frame, keeping all color channels (RGB).")
    st.write("The extracted ROI is then appended to the `frames` list.")

    st.write("8. `cap.release()`")
    st.write("This line releases the video capture object to free up system resources after processing all frames.")

    st.write("9. `mean = tf.math.reduce_mean(frames)`")
    st.write("This line calculates the mean value of all frames in the `frames` list using TensorFlow's `tf.math.reduce_mean()` function.")

    st.write("10. `std = tf.math.reduce_std(tf.cast(frames, tf.float32))`")
    st.write("This line calculates the standard deviation of the `frames` list after converting it to a TensorFlow float32 tensor.")

    st.write("11. `return tf.cast((frames - mean), tf.float32) / std`")
    st.write("This line returns the normalized frames as a TensorFlow float32 tensor.")
    st.write("It subtracts the mean value from each frame, converts the frames to float32 data type, and then divides by the standard deviation to normalize the data.")

st.divider()

code = '''char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

print(
    f"The vocabulary is: {char_to_num.get_vocabulary()} "
    f"(size ={char_to_num.vocabulary_size()})"
)'''
st.code(code, language='python')

with st.expander("Converting Characters to Numbers and Numbers to Characters"):
    st.write("The code defines two TensorFlow string lookup layers to convert characters to numbers and vice versa.")
    st.write("1. `char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")`")
    st.write("This line creates a string lookup layer called `char_to_num`.")
    st.write("The layer is initialized with a vocabulary (a set of characters) specified by the `vocab` variable.")
    st.write("The `oov_token` parameter is set to an empty string, indicating that any out-of-vocabulary character will be mapped to an empty token.")

    st.write("2. `num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)`")
    st.write("This line creates another string lookup layer called `num_to_char`.")
    st.write("The layer's vocabulary is set to the same vocabulary as `char_to_num` using `char_to_num.get_vocabulary()`.")
    st.write("The `oov_token` parameter is set to an empty string, meaning that any out-of-vocabulary number will be mapped to an empty token.")
    st.write("The `invert=True` parameter allows the layer to convert numbers back to characters, effectively inverting the mapping.")

    st.write("3. `print(f'The vocabulary is: {char_to_num.get_vocabulary()} (size ={char_to_num.vocabulary_size()})')`")
    st.write("This line prints the vocabulary of the `char_to_num` layer and its size.")
    st.write("The `get_vocabulary()` function retrieves the vocabulary set from the `char_to_num` layer.")
    st.write("The `vocabulary_size()` function returns the number of unique characters in the vocabulary.")
    st.write("The output will display the vocabulary and its size in the format: 'The vocabulary is: [characters] (size = [vocabulary size])'.")

st.divider()

code = '''def load_alignments(path:str) -> List[str]: 
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]'''
st.code(code, language='python')

with st.expander("Load Alignments and Convert Characters Into Numbers"):
    st.write("The code defines a function called `load_alignments` that takes a file path as input.")
    st.write("1. `def load_alignments(path:str) -> List[str]:`")
    st.write("This line defines the function `load_alignments` that takes a single argument `path`, representing the file path of the alignments file.")
    st.write("The function returns a list of strings.")

    st.write("2. `with open(path, 'r') as f:`")
    st.write("This line opens the file specified by `path` in read mode ('r') using a context manager.")
    st.write("The file content is accessible within the indented block of code.")
    
    st.write("3. `lines = f.readlines()`")
    st.write("This line reads all the lines from the opened file and stores them in a list called `lines`.")
    
    st.write("4. `tokens = []`")
    st.write("This line initializes an empty list called `tokens` to store the alignment tokens after processing.")
    
    st.write("5. `for line in lines:`")
    st.write("This line starts a loop that iterates over each line in the `lines` list.")
    st.write("The loop processes each line of the alignments file.")
    
    st.write("6. `line = line.split()`")
    st.write("This line splits the current line into a list of elements using whitespaces as the separator.")
    st.write("The split elements are stored in the `line` variable.")
    
    st.write("7. `if line[2] != 'sil':`")
    st.write("This line checks if the third element in the `line` list is not equal to the string 'sil'.")
    st.write("If true, it means the token is not 'sil' (silence).")
    
    st.write("8. `tokens = [*tokens, ' ', line[2]]`")
    st.write("This line appends a space (' ') and the third element of the `line` list (the alignment token) to the `tokens` list.")
    st.write("The alignment tokens are collected in the `tokens` list, separated by spaces.")
    
    st.write("9. `return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]`")
    st.write("This line returns the alignment tokens as a TensorFlow tensor of integers.")
    st.write("The `tf.strings.unicode_split()` function splits the tokens into individual characters.")
    st.write("The `tf.reshape()` function reshapes the resulting characters into a 1D tensor.")
    st.write("The `char_to_num` layer (defined elsewhere) is used to convert the characters to their corresponding integers.")
    st.write("The `[1:]` indexing is used to remove the first element of the tensor (the space added at the beginning).")

st.divider()

code = '''def load_data(path: str): 
    path = bytes.decode(path.numpy())
    #file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('data','alignments','s1',f'{file_name}.align')
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments'''
st.code(code, language='python')

with st.expander("Loads Data: Video Frames and Alignment Functions and Joins Them"):
    st.write("The code defines a function called `load_data` that takes a file path as input.")
    st.write("1. `def load_data(path: str):`")
    st.write("This line defines the function `load_data` that takes a single argument `path`, representing the file path.")
    st.write("The function returns two values: `frames` and `alignments`.")

    st.write("2. `path = bytes.decode(path.numpy())`")
    st.write("This line decodes the given file path from bytes to a string using the `decode()` method.")
    st.write("The `path` variable is updated to hold the decoded string value.")
    
    st.write("3. `file_name = path.split('\\')[-1].split('.')[0]`")
    st.write("This line extracts the file name from the given file path.")
    st.write("The file path is split using the backslash '\\' as the separator to get the filename and the extension.")
    st.write("The `[0]` indexing is used to retrieve only the file name part, excluding the extension.")
    
    st.write("4. `video_path = os.path.join('data','s1',f'{file_name}.mpg')`")
    st.write("This line creates the video path by joining different parts using `os.path.join()`.")
    st.write("The video path is constructed as `'data/s1/[file_name].mpg'`, where `[file_name]` is the name of the video file.")
    
    st.write("5. `alignment_path = os.path.join('data','alignments','s1',f'{file_name}.align')`")
    st.write("This line creates the alignment path using `os.path.join()`.")
    st.write("The alignment path is constructed as `'data/alignments/s1/[file_name].align'`, where `[file_name]` is the name of the alignment file.")
    
    st.write("6. `frames = load_video(video_path)`")
    st.write("This line calls the `load_video` function with the `video_path` as input.")
    st.write("The function returns the frames of the video, and they are assigned to the `frames` variable.")
    
    st.write("7. `alignments = load_alignments(alignment_path)`")
    st.write("This line calls the `load_alignments` function with the `alignment_path` as input.")
    st.write("The function returns the alignments data, and it is assigned to the `alignments` variable.")
    
    st.write("8. `return frames, alignments`")
    st.write("This line returns two values: `frames` and `alignments` as the output of the `load_data` function.")
    st.write("These two variables hold the video frames and alignment data that will be used for further processing.")

st.divider()

code = '''tf.convert_to_tensor(test_path).numpy().decode('utf-8').split('\\')[-1].split('.')[0]'''
st.code(code, language='python')

with st.expander("Convert and Extract File Name"):
    st.write("The code converts a TensorFlow tensor to a string and extracts the file name from a given file path.")
    st.write("1. `tf.convert_to_tensor(test_path)`")
    st.write("This line converts the variable `test_path` to a TensorFlow tensor.")
    st.write("The `tf.convert_to_tensor()` function is used to ensure that the input is in a format compatible with TensorFlow.")
    
    st.write("2. `.numpy()`")
    st.write("This line converts the TensorFlow tensor back to a NumPy array.")
    st.write("The `numpy()` method is used to extract the value from the TensorFlow tensor as a NumPy array.")
    
    st.write("3. `.decode('utf-8')`")
    st.write("This line decodes the NumPy array from bytes to a UTF-8 encoded string.")
    st.write("The `decode()` method is applied with the `'utf-8'` encoding to convert the bytes to a human-readable string.")
    
    st.write("4. `.split('\\')`")
    st.write("This line splits the string using the backslash `'\\'` as the separator.")
    st.write("It separates the path into different parts.")
    
    st.write("5. `[-1]`")
    st.write("This indexing retrieves the last element from the list of parts obtained after splitting.")
    
    st.write("6. `.split('.')`")
    st.write("This line splits the string again, using the dot `'.'` as the separator.")
    st.write("It separates the file name from its extension.")
    
    st.write("7. `[0]`")
    st.write("This indexing retrieves the first element from the list of parts obtained after the second split.")
    st.write("The first element is the file name without the extension.")
    
    st.write("The code finally returns the extracted file name.")

st.divider()

code = '''frames, alignments = load_data(tf.convert_to_tensor(test_path))'''
st.code(code, language='python')

with st.expander("Load Data"):
    st.write("The code loads data using the function `load_data` with a given file path.")
    st.write("1. `frames, alignments = load_data(tf.convert_to_tensor(test_path))`")
    st.write("This line calls the `load_data` function with the input `tf.convert_to_tensor(test_path)`.")
    st.write("The `tf.convert_to_tensor(test_path)` converts the variable `test_path` to a TensorFlow tensor.")
    st.write("The function `load_data` takes this tensor as input and returns two values: `frames` and `alignments`.")
    
    st.write("2. `frames`")
    st.write("After calling the function, `frames` will store the video frames obtained from the `load_data` function.")
    st.write("The video frames are the frames of the video located at the path provided.")
    
    st.write("3. `alignments`")
    st.write("After calling the function, `alignments` will store the alignment data obtained from the `load_data` function.")
    st.write("The alignments are the data associated with the video's alignment, which may contain information like timestamps, transcriptions, or other relevant details.")
    
    st.write("In summary, this line loads video frames and alignment data by calling the `load_data` function with the provided file path.")
    st.write("The function returns the frames and alignments as output, which can be used for further processing or analysis.")

st.divider()

code = '''tf.strings.reduce_join([bytes.decode(x) for x in num_to_char(alignments.numpy()).numpy()])'''
st.code(code, language='python')

with st.expander("Decode Alignments"):
    st.write("The code decodes the alignment data obtained from a TensorFlow tensor.")
    st.write("1. `tf.strings.reduce_join([bytes.decode(x) for x in num_to_char(alignments.numpy()).numpy()])`")
    st.write("This line calls the `num_to_char` function with the input `alignments.numpy()`.")
    st.write("The `alignments.numpy()` converts the tensor `alignments` to a NumPy array.")
    st.write("The `num_to_char` function takes this NumPy array as input and converts the alignment numbers back to characters.")
    st.write("The `bytes.decode(x)` decodes each character (in bytes format) to a UTF-8 encoded string.")
    st.write("A list comprehension is used to iterate through each character in the NumPy array and decode them.")
    st.write("The result is a list of characters represented as UTF-8 encoded strings.")
    
    st.write("2. `tf.strings.reduce_join()`")
    st.write("This line uses the TensorFlow function `tf.strings.reduce_join()` to concatenate all the decoded characters.")
    st.write("The `reduce_join()` function joins the strings in the list into a single string.")
    st.write("The resulting string is the decoded version of the alignment data.")
    
    st.write("In summary, this line decodes the alignment data from a TensorFlow tensor.")
    st.write("It first converts the alignment numbers to characters using the `num_to_char` function.")
    st.write("Then, it decodes each character to a UTF-8 encoded string.")
    st.write("Finally, it joins all the decoded characters into a single string, representing the decoded alignment data.")
    st.write("The decoded alignment data can be used for further analysis or display.")

st.divider()

code = '''def mappable_function(path:str) ->List[str]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result'''
st.code(code, language='python')

import streamlit as st

with st.expander("Wrapping inside a Mappable Function (Better With TensorFlow Pipelines)"):
    st.write("The code defines a function called `mappable_function` that takes a file path as input.")
    st.write("1. `def mappable_function(path:str) -> List[str]:`")
    st.write("This line defines the function `mappable_function` that takes a single argument `path`, representing the file path.")
    st.write("The function returns a result of type `(tf.float32, tf.int64)`.")

    st.write("2. `result = tf.py_function(load_data, [path], (tf.float32, tf.int64))`")
    st.write("This line calls the TensorFlow function `tf.py_function()` to apply a Python function (`load_data`) to the input `path`.")
    st.write("The `load_data` function is applied with the file path as input.")
    st.write("The result is assigned to the variable `result`.")
    
    st.write("3. `return result`")
    st.write("This line returns the `result`, which is a tuple of two TensorFlow tensors: one of data type `tf.float32` and the other of data type `tf.int64`.")
    st.write("The function `mappable_function` effectively applies the `load_data` function to the given `path` and returns the resulting TensorFlow tensors.")
    
    st.write("In summary, this function allows the `load_data` function to be applied to a file path and returns the result as two TensorFlow tensors.")
    st.write("The `mappable_function` can be used in the context of TensorFlow data mapping, allowing efficient processing of data in TensorFlow datasets.")

st.divider()

code = '''data = tf.data.Dataset.list_files('./data/s1/*.mpg')
data = data.shuffle(500, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(2, padded_shapes=([75,None,None,None],[40]))
data = data.prefetch(tf.data.AUTOTUNE)
# Added for split 
train = data.take(450)
test = data.skip(450)'''
st.code(code, language='python')

with st.expander("Create and Process Dataset (PipeLine)"):
    st.write("The code creates a TensorFlow dataset and applies several operations to it.")
    st.write("1. `data = tf.data.Dataset.list_files('./data/s1/*.mpg')`")
    st.write("This line creates a TensorFlow dataset using `tf.data.Dataset.list_files()`.")
    st.write("It searches for files with the extension `.mpg` in the directory `./data/s1/` and creates a dataset with their paths.")
    
    st.write("2. `data = data.shuffle(500, reshuffle_each_iteration=False)`")
    st.write("This line shuffles the dataset randomly to increase data randomness and reduce potential biases.")
    st.write("The `reshuffle_each_iteration=False` argument means that the dataset will not be reshuffled at each iteration.")
    st.write("The dataset will be shuffled using a buffer size of 500.")
    
    st.write("3. `data = data.map(mappable_function)`")
    st.write("This line applies the function `mappable_function` to each element of the dataset.")
    st.write("The `map()` function is used to apply a transformation to each element of the dataset.")
    
    st.write("4. `data = data.padded_batch(2, padded_shapes=([75,None,None,None],[40]))`")
    st.write("This line creates batches of 2 elements from the dataset.")
    st.write("Each batch is padded to a specific shape.")
    st.write("The first element of the padded shape is `[75, None, None, None]`, and the second element is `[40]`.")
    
    st.write("5. `data = data.prefetch(tf.data.AUTOTUNE)`")
    st.write("This line prefetches the data to overlap data preprocessing and model execution.")
    st.write("The `tf.data.AUTOTUNE` argument allows TensorFlow to determine the optimal number of elements to prefetch automatically.")
    
    st.write("6. `train = data.take(450)`")
    st.write("This line creates a new dataset `train` by taking the first 450 elements from the original dataset `data`.")
    st.write("This dataset `train` will be used for training the model.")
    
    st.write("7. `test = data.skip(450)`")
    st.write("This line creates a new dataset `test` by skipping the first 450 elements from the original dataset `data`.")
    st.write("This dataset `test` will be used for testing the model.")
    
    st.write("In summary, this code creates a TensorFlow dataset from files in the `./data/s1/` directory.")
    st.write("It shuffles the dataset, applies a mapping function, creates batches with padding, and prefetches the data.")
    st.write("The resulting dataset is split into training (`train`) and testing (`test`) datasets.")
    st.write("These datasets can be used for training and evaluating a machine learning model.")

st.divider()

code = '''imageio.mimsave('./animation.gif', val[0][0], fps=10)'''
st.code(code, language='python')

with st.expander("Converting Numpy Array to a GIF"):
    st.write("The code creates an animation GIF from a 3D array.")
    st.write("1. `imageio.mimsave('./animation.gif', val[0][0], fps=10)`")
    st.write("This line calls the `imageio.mimsave()` function to save an animation GIF.")
    st.write("The animation GIF will be saved with the filename `./animation.gif`.")
    st.write("The input data for the GIF is `val[0][0]`, which should be a 3D array representing the frames of the animation.")
    st.write("The `fps=10` argument sets the frames per second for the animation to 10, which means it will play at 10 frames per second.")
    
    st.write("The `imageio.mimsave()` function takes the 3D array `val[0][0]` and creates an animation GIF.")
    st.write("The resulting GIF file will be saved as `animation.gif` in the current directory.")
    
    st.write("In summary, this line creates an animation GIF from a 3D array and saves it as `animation.gif` with a frame rate of 10 frames per second.")
    st.write("The resulting GIF can be used to visualize the animation created from the input 3D array.")

st.divider()

code = '''tf.strings.reduce_join([num_to_char(word) for word in val[1][0]])'''
st.code(code, language='python')

with st.expander("Pre-Processed Annotations or Alignments"):
    st.write("The code reduces and joins characters from a TensorFlow tensor representing words.")
    st.write("1. `tf.strings.reduce_join([num_to_char(word) for word in val[1][0]])`")
    st.write("This line creates a list comprehension to iterate through each word in the 2D array `val[1][0]`.")
    st.write("For each word, it calls the function `num_to_char()` to convert the numbers in the word to characters.")
    st.write("The `num_to_char()` function converts each number to its corresponding character using a vocabulary.")
    st.write("The result is a list of strings where each string represents a word with characters instead of numbers.")
    
    st.write("2. `tf.strings.reduce_join()`")
    st.write("This line uses the TensorFlow function `tf.strings.reduce_join()` to concatenate all the words in the list.")
    st.write("The `reduce_join()` function joins the strings in the list into a single string.")
    st.write("The resulting string is the reduced and joined version of all the words.")
    
    st.write("In summary, this line reduces and joins characters from a TensorFlow tensor representing words.")
    st.write("It converts the numbers in each word to characters using the `num_to_char()` function.")
    st.write("Then, it joins all the words into a single string, representing the reduced and joined version of the words.")
    st.write("The resulting string can be used for further analysis or display.")

st.divider()

code = '''model = Sequential()
model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(Conv3D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(Conv3D(75, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(TimeDistributed(Flatten()))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))'''
st.code(code, language='python')

with st.expander("Deep Neural Network"):
    st.write("The code defines a Sequential model using the Keras API.")
    st.write("The model architecture consists of several layers for a specific task, such as classification.")
    st.write("1. `model = Sequential()`")
    st.write("This line creates a new Keras Sequential model.")
    st.write("The model is an empty container where layers can be added sequentially.")
    
    st.write("2. `model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))`")
    st.write("This line adds a 3D convolutional layer to the model with 128 filters.")
    st.write("The kernel size for the convolutional layer is 3x3x3.")
    st.write("The input shape of the layer is (75, 46, 140, 1), representing the dimensions of the input data.")
    st.write("The padding is set to 'same', which pads the input with zeros to maintain the same output size.")
    
    st.write("3. `model.add(Activation('relu'))`")
    st.write("This line adds a ReLU activation function to the model.")
    st.write("The ReLU activation function applies the element-wise Rectified Linear Unit function, which introduces non-linearity to the model.")
    
    st.write("4. `model.add(MaxPool3D((1, 2, 2)))`")
    st.write("This line adds a 3D max-pooling layer to the model.")
    st.write("The max-pooling layer performs a 3D pooling operation with a pool size of (1, 2, 2).")
    st.write("This operation reduces the spatial dimensions of the data by half along the second and third axes.")
    
    st.write("5. `model.add(Conv3D(256, 3, padding='same'))`")
    st.write("This line adds another 3D convolutional layer to the model with 256 filters.")
    st.write("The kernel size for this convolutional layer is also 3x3x3.")
    st.write("The padding is set to 'same', maintaining the same output size.")
    
    st.write("6. `model.add(Activation('relu'))`")
    st.write("This line adds another ReLU activation function.")
    
    st.write("7. `model.add(MaxPool3D((1, 2, 2)))`")
    st.write("This line adds another 3D max-pooling layer with the same pool size of (1, 2, 2).")
    
    st.write("8. `model.add(Conv3D(75, 3, padding='same'))`")
    st.write("This line adds a third 3D convolutional layer to the model with 75 filters.")
    st.write("The kernel size is 3x3x3, and the padding is set to 'same'.")
    
    st.write("9. `model.add(Activation('relu'))`")
    st.write("This line adds another ReLU activation function.")
    
    st.write("10. `model.add(MaxPool3D((1, 2, 2)))`")
    st.write("This line adds another 3D max-pooling layer with the same pool size of (1, 2, 2).")
    
    st.write("11. `model.add(TimeDistributed(Flatten()))`")
    st.write("This line adds a TimeDistributed layer with the Flatten function.")
    st.write("The TimeDistributed layer applies the Flatten operation to each time step of the input sequence.")
    st.write("This operation flattens the input into a 1D array, suitable for further processing.")
    
    st.write("12. `model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))`")
    st.write("This line adds a bidirectional LSTM layer to the model with 128 units.")
    st.write("The LSTM layer is initialized with the 'Orthogonal' kernel initializer.")
    st.write("The return_sequences parameter is set to True, which means the LSTM layer returns the full sequence of outputs.")
    
    st.write("13. `model.add(Dropout(0.5))`")
    st.write("This line adds a Dropout layer to the model with a dropout rate of 0.5.")
    st.write("The Dropout layer randomly sets a fraction of input units to 0 during training, which helps prevent overfitting.")
    
    st.write("14. `model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))`")
    st.write("This line adds another bidirectional LSTM layer with the same configuration as the previous one.")
    
    st.write("15. `model.add(Dropout(0.5))`")
    st.write("This line adds another Dropout layer with the same dropout rate of 0.5.")
    
    st.write("16. `model.add(Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'))`")
    st.write("This line adds a Dense output layer to the model.")
    st.write("The number of units in this layer is equal to the vocabulary size of the character-to-number mapping plus one.")
    st.write("The kernel initializer is set to 'he_normal', and the activation function is 'softmax'.")
    st.write("The softmax activation function is commonly used in multi-class classification tasks.")
    
    st.write("In summary, this code creates a Sequential model with a specific architecture using the Keras API.")
    st.write("The model contains several layers, including 3D convolutional layers, activation functions, max-pooling layers, bidirectional LSTM layers, dropout layers, and a dense output layer.")
    st.write("Each layer is added sequentially to the model.")
    st.write("The model can be compiled and trained with appropriate data to perform a specific task, such as classification or regression.")

st.divider()

code = '''yhat = model.predict(val[0])'''
st.code(code, language='python')

with st.expander("Make Predictions"):
    st.write("The code is for making predictions using the trained model on the validation data.")
    st.write("1. `yhat = model.predict(val[0])`")
    st.write("This line uses the trained model to make predictions on the validation data.")
    st.write("The input to the model is `val[0]`, which contains the frames of the validation data.")
    st.write("The `predict` function applies the trained model to the input data and returns the predictions.")
    st.write("The output `yhat` will contain the model's predictions for the validation data.")
    
    st.code("yhat = model.predict(val[0])")
    
    st.write("In summary, this code makes predictions using the trained Keras model on the validation data.")
    st.write("The predictions can be used to evaluate the model's performance or for any other inference tasks.")

st.divider()

code = '''def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss'''
st.code(code, language='python')

with st.expander("CTC Loss Function - Automatic Speech Recognition Model - TensorFlow"):
    st.write("The code defines a custom CTC (Connectionist Temporal Classification) loss function for training the model.")
    st.write("The CTC loss is often used in sequence-to-sequence tasks, such as speech recognition and handwriting recognition.")
    
    st.write("1. `def CTCLoss(y_true, y_pred):`")
    st.write("This line defines the custom CTC loss function named `CTCLoss`.")
    st.write("The function takes two arguments: `y_true` - the ground truth labels, and `y_pred` - the predicted output from the model.")
    
    st.write("2. `batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')`")
    st.write("This line calculates the batch size by getting the number of sequences in the ground truth labels `y_true`.")
    st.write("It uses TensorFlow's `tf.shape` function to get the shape of `y_true` and then converts it to an int64 data type using `tf.cast`.")
    
    st.write("3. `input_length = tf.cast(tf.shape(y_pred)[1], dtype='int64')`")
    st.write("This line calculates the input length by getting the number of time steps in the predicted output `y_pred`.")
    st.write("It uses TensorFlow's `tf.shape` function to get the shape of `y_pred` and then converts it to an int64 data type using `tf.cast`.")
    
    st.write("4. `label_length = tf.cast(tf.shape(y_true)[1], dtype='int64')`")
    st.write("This line calculates the label length by getting the number of time steps in the ground truth labels `y_true`.")
    st.write("It uses TensorFlow's `tf.shape` function to get the shape of `y_true` and then converts it to an int64 data type using `tf.cast`.")
    
    st.write("5. `input_length = input_length * tf.ones(shape=(batch_len, 1), dtype='int64')`")
    st.write("This line creates a tensor `input_length` of shape (batch_len, 1) where each element is set to the value of `input_length`.")
    st.write("It is done to ensure that each sequence in the batch has the same input length.")
    
    st.write("6. `label_length = label_length * tf.ones(shape=(batch_len, 1), dtype='int64')`")
    st.write("This line creates a tensor `label_length` of shape (batch_len, 1) where each element is set to the value of `label_length`.")
    st.write("It is done to ensure that each sequence in the batch has the same label length.")
    
    st.write("7. `loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)`")
    st.write("This line calculates the CTC loss using TensorFlow's `ctc_batch_cost` function.")
    st.write("The `ctc_batch_cost` function computes the CTC loss for each element in the batch.")
    
    st.write("8. `return loss`")
    st.write("This line returns the computed CTC loss as the output of the function.")
    
    st.write("In summary, the `CTCLoss` function computes the CTC loss for a batch of sequences.")
    st.write("It ensures that all sequences in the batch have the same input and label length, as required by the CTC loss computation.")
    st.write("The CTC loss is a crucial component in training models for sequence-to-sequence tasks, where the alignment between input and output sequences is not known.")

st.divider()

code = '''class ProduceExample(tf.keras.callbacks.Callback): 
    def __init__(self, dataset) -> None: 
        self.dataset = dataset.as_numpy_iterator()
    
    def on_epoch_end(self, epoch, logs=None) -> None:
        data = self.dataset.next()
        yhat = self.model.predict(data[0])
        decoded = tf.keras.backend.ctc_decode(yhat, [75,75], greedy=False)[0][0].numpy()
        for x in range(len(yhat)):           
            print('Original:', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8'))
            print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
            print('~'*100)'''
st.code(code, language='python')

with st.expander("Output the Original Align File and the Prediction"):
    st.write("The code defines a custom callback class named `ProduceExample`.")
    st.write("This class is used during the training of the model to print and compare original and predicted sequences after each epoch.")
    
    st.write("1. `class ProduceExample(tf.keras.callbacks.Callback):`")
    st.write("This line defines the custom callback class named `ProduceExample` that extends `tf.keras.callbacks.Callback`.")
    st.write("The class is responsible for printing and comparing original and predicted sequences after each epoch.")
    
    st.write("2. `def __init__(self, dataset) -> None:`")
    st.write("This line defines the constructor of the `ProduceExample` class.")
    st.write("The constructor takes a single argument `dataset`, which is expected to be a TensorFlow dataset.")
    st.write("It initializes the class with the given dataset as `self.dataset`.")
    
    st.write("3. `def on_epoch_end(self, epoch, logs=None) -> None:`")
    st.write("This line defines the `on_epoch_end` method of the `ProduceExample` class.")
    st.write("The method is called by TensorFlow after each epoch of training is completed.")
    
    st.write("4. `data = self.dataset.next()`")
    st.write("This line retrieves the next batch of data from the dataset.")
    st.write("The `next()` method of the dataset iterator is used to get the data for the current epoch.")
    
    st.write("5. `yhat = self.model.predict(data[0])`")
    st.write("This line uses the trained model to make predictions on the input data `data[0]`.")
    st.write("The predicted output is stored in the variable `yhat`.")
    
    st.write("6. `decoded = tf.keras.backend.ctc_decode(yhat, [75, 75], greedy=False)[0][0].numpy()`")
    st.write("This line uses TensorFlow's `ctc_decode` function to decode the predicted output `yhat`.")
    st.write("The `ctc_decode` function performs CTC decoding to convert the predicted probabilities into sequences of characters.")
    st.write("The parameter `[75, 75]` represents the input length for decoding, and `greedy=False` indicates that the decoding is not greedy.")
    st.write("The decoded sequences are stored in the variable `decoded` as a NumPy array.")
    
    st.write("7. `for x in range(len(yhat)):`")
    st.write("This line starts a loop that iterates over each sample in the batch.")
    st.write("The loop runs for the number of samples in the batch.")
    
    st.write("8. `print('Original:', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8'))`")
    st.write("This line prints the original sequence for the current sample.")
    st.write("It uses the `num_to_char` function to convert the numerical labels `data[1][x]` back to characters.")
    st.write("The `tf.strings.reduce_join` function joins the characters into a single string.")
    
    st.write("9. `print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))`")
    st.write("This line prints the predicted sequence for the current sample.")
    st.write("It uses the `num_to_char` function to convert the numerical labels `decoded[x]` back to characters.")
    st.write("The `tf.strings.reduce_join` function joins the characters into a single string.")
    
    st.write("10. `print('~'*100)`")
    st.write("This line prints a separator line (a row of 100 tildes) to separate the output for each sample.")
    
    st.write("In summary, the `ProduceExample` class is a custom callback used during model training to print and compare original and predicted sequences after each epoch.")
    st.write("It helps in visualizing the model's performance and understanding how well it is predicting the target sequences.")

st.divider()

code = '''model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)'''
st.code(code, language='python')

with st.expander("Model Compilation"):
    st.write("The code compiles the model before training.")
    
    st.write("1. `model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)`")
    st.write("This line compiles the model for training.")
    st.write("It specifies the optimizer and the loss function to be used during training.")
    
    st.write("2. `optimizer=Adam(learning_rate=0.0001)`")
    st.write("The `Adam` optimizer is used for updating the model's weights during training.")
    st.write("The `learning_rate=0.0001` sets the learning rate for the optimizer, which controls the step size during weight updates.")
    st.write("A smaller learning rate typically results in slower convergence but can provide better results.")
    
    st.write("3. `loss=CTCLoss`")
    st.write("The `CTCLoss` function is used as the loss function for training the model.")
    st.write("CTC stands for Connectionist Temporal Classification, and it is commonly used for sequence-to-sequence tasks like speech recognition and text generation.")
    st.write("The CTC loss function helps in training sequence models by handling variable-length input and output sequences.")
    
    st.write("In summary, the model compilation step sets the optimizer and loss function for the model, preparing it for the training process.")
    st.write("With the appropriate optimizer and loss function, the model is ready to be trained on the preprocessed data.")

st.divider()

code = '''checkpoint_callback = ModelCheckpoint(os.path.join('models','checkpoint'), monitor='loss', save_weights_only=True) 
schedule_callback = LearningRateScheduler(scheduler)
example_callback = ProduceExample(test)
model.fit(train, validation_data=test, epochs=100, callbacks=[checkpoint_callback, schedule_callback, example_callback])'''
st.code(code, language='python')

with st.expander("Model Training with Callbacks"):
    st.write("The code trains the model using callbacks for monitoring and custom behavior during training.")
    
    st.write("1. `checkpoint_callback = ModelCheckpoint(os.path.join('models', 'checkpoint'), monitor='loss', save_weights_only=True)`")
    st.write("This line creates a `ModelCheckpoint` callback.")
    st.write("The `ModelCheckpoint` callback saves the model's weights during training.")
    st.write("The `monitor='loss'` parameter specifies that the callback will monitor the training loss to determine when to save the model weights.")
    st.write("The `save_weights_only=True` parameter indicates that only the model's weights will be saved, not the entire model.")
    st.write("The saved model weights will be stored in the 'models/checkpoint' directory.")
    
    st.write("2. `schedule_callback = LearningRateScheduler(scheduler)`")
    st.write("This line creates a `LearningRateScheduler` callback.")
    st.write("The `LearningRateScheduler` callback is used to schedule the learning rate during training.")
    st.write("The `scheduler` function will be used to adjust the learning rate based on the current epoch number.")
    
    st.write("3. `example_callback = ProduceExample(test)`")
    st.write("This line creates a `ProduceExample` callback.")
    st.write("The `ProduceExample` callback is a custom callback created earlier.")
    st.write("It will be used to print and compare original and predicted sequences after each epoch.")
    st.write("The callback is constructed with the test dataset `test`.")
    
    st.write("4. `model.fit(train, validation_data=test, epochs=100, callbacks=[checkpoint_callback, schedule_callback, example_callback])`")
    st.write("This line trains the model using the `fit` method.")
    st.write("It trains the model on the `train` dataset and validates it on the `test` dataset.")
    st.write("The `epochs=100` parameter specifies the number of training epochs.")
    st.write("The `callbacks` parameter is a list containing the previously created callbacks.")
    st.write("During training, the model will save weights using the `ModelCheckpoint` callback, adjust the learning rate with the `LearningRateScheduler` callback, and print example outputs using the `ProduceExample` callback.")
    
    st.write("In summary, the model training step uses callbacks to monitor and customize the training process.")
    st.write("The `ModelCheckpoint` callback saves the model's weights, the `LearningRateScheduler` callback adjusts the learning rate, and the custom `ProduceExample` callback prints and compares example sequences after each epoch.")

st.divider()

code = '''test_data = test.as_numpy_iterator()
sample = test_data.next()
yhat = model.predict(sample[0])
print('~'*100, 'REAL TEXT')
[tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in sample[1]
decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75,75], greedy=True)[0][0].numpy()
print('~'*100, 'PREDICTIONS')
[tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded]'''
st.code(code, language='python')

with st.expander("Making Predictions and Comparing"):
    st.write("The code makes predictions using the trained model and compares the real text with the predictions.")
    
    st.write("1. `test_data = test.as_numpy_iterator()`")
    st.write("This line converts the test dataset `test` into a numpy iterator `test_data`.")
    st.write("It allows us to access individual samples from the dataset for making predictions and comparisons.")
    
    st.write("2. `sample = test_data.next()`")
    st.write("This line retrieves the next sample from the test dataset using the numpy iterator `test_data`.")
    st.write("The `sample` now contains a batch of two samples along with their ground truth text.")
    
    st.write("3. `yhat = model.predict(sample[0])`")
    st.write("This line uses the trained model to make predictions on the input data `sample[0]`.")
    st.write("The `yhat` variable now contains the predicted sequences for the given batch.")
    
    st.write("4. `print('~'*100, 'REAL TEXT')`")
    st.write("This line prints a separator to separate the real text from the predictions in the output.")
    
    st.write("5. `[tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in sample[1]]`")
    st.write("This line processes and prints the ground truth text for the batch.")
    st.write("It iterates through each sentence in the `sample[1]` and converts the numeric values back to characters using the `num_to_char` function.")
    st.write("The `tf.strings.reduce_join` function then joins the characters into strings, representing the real text.")
    
    st.write("6. `decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75,75], greedy=True)[0][0].numpy()`")
    st.write("This line decodes the predictions using the CTC decoding method.")
    st.write("The `yhat` contains the model's predictions, and the `input_length=[75,75]` specifies the length of input sequences to be used for decoding.")
    st.write("The `greedy=True` parameter indicates that the decoding should use the greedy approach, which is faster but may not produce the most accurate results.")
    st.write("The `decoded` variable now contains the decoded sequences as numeric values.")
    
    st.write("7. `print('~'*100, 'PREDICTIONS')`")
    st.write("This line prints a separator to separate the predictions from the real text in the output.")
    
    st.write("8. `[tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded]`")
    st.write("This line processes and prints the predicted text for the batch.")
    st.write("It iterates through each sentence in the `decoded` and converts the numeric values back to characters using the `num_to_char` function.")
    st.write("The `tf.strings.reduce_join` function then joins the characters into strings, representing the predicted text.")
    
    st.write("In summary, the code makes predictions using the trained model and compares the real text with the predictions.")
    st.write("It prints the real text and the predicted text side by side to compare the accuracy of the model's predictions.")

colored_header(
    label="StreamLip App",
    description="Where The Magic Show Up",
    color_name="violet-70",
)

code = '''options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)'''
st.code(code, language='python')

with st.expander("Select Video"):
    st.write("The code allows the user to select a video from the available options.")
    
    st.write("1. `options = os.listdir(os.path.join('..', 'data', 's1'))`")
    st.write("This line gets a list of files in the 'data/s1' directory using the `os.listdir` function.")
    st.write("The list `options` now contains the names of all files present in the 'data/s1' directory.")
    
    st.write("2. `selected_video = st.selectbox('Choose video', options)`")
    st.write("This line creates a selectbox widget using Streamlit's `st.selectbox` function.")
    st.write("The selectbox allows the user to choose a video from the available options.")
    st.write("The first parameter `'Choose video'` is the label displayed above the selectbox, prompting the user to make a selection.")
    st.write("The second parameter `options` is the list of video names obtained in the previous step.")
    st.write("The selected video will be stored in the variable `selected_video`.")
    
    st.write("In summary, the code creates a selectbox that allows the user to choose a video from the available options in the 'data/s1' directory.")
    st.write("The selected video name will be stored in the `selected_video` variable for further processing.")

st.divider()

code = '''col1, col2 = st.columns(2)

if options: 
 
    with col1: 
        st.text('Converted video from MPG to MP4')
        file_path = os.path.join('..','data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)'''
st.code(code, language='python')

with st.expander("Rendering Video"):
    st.write("The code renders a video on the Streamlit app based on the selected video.")

    st.write("1. `col1, col2 = st.columns(2)`")
    st.write("This line creates two columns (`col1` and `col2`) using Streamlit's `st.columns` function.")
    st.write("The `2` parameter specifies that we want two equal-width columns in the app layout.")

    st.write("2. `if options:`")
    st.write("This line checks if there are any video options available.")
    st.write("If there are no options (i.e., no videos present), this block of code will not be executed.")

    st.write("3. `with col1:`")
    st.write("This line starts a context manager for rendering content inside `col1`.")
    st.write("All content within the `with col1:` block will be displayed in the left column of the app.")

    st.write("4. `st.text('Converted video from MPG to MP4')`")
    st.write("This line displays a text message in the left column to indicate that the video is being converted from MPG to MP4.")

    st.write("5. `file_path = os.path.join('..', 'data', 's1', selected_video)`")
    st.write("This line constructs the full file path of the selected video based on the `selected_video` variable.")

    st.write("6. `os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')`")
    st.write("This line uses the `os.system` function to run the `ffmpeg` command-line tool.")
    st.write("The `ffmpeg` command converts the selected video from MPG to MP4 format using the specified codec.")
    st.write("The converted MP4 video will be saved as 'test_video.mp4' in the same directory as the app.")

    st.write("7. `video = open('test_video.mp4', 'rb')`")
    st.write("This line opens the converted MP4 video file in binary read mode (`'rb'`).")

    st.write("8. `video_bytes = video.read()`")
    st.write("This line reads the binary content of the video file into the `video_bytes` variable.")

    st.write("9. `st.video(video_bytes)`")
    st.write("This line displays the video in the left column of the app using Streamlit's `st.video` function.")
    st.write("The video is rendered based on the binary content stored in `video_bytes`.")

    st.write("In summary, the code renders a video in the left column of the app.")
    st.write("First, it converts the selected video from MPG to MP4 format using `ffmpeg`.")
    st.write("Then, it reads the converted MP4 video file and displays it using `st.video`.")
    st.write("If there are no video options available (`options` is empty), this block of code will not be executed.")

st.divider()

code = '''with col2: 
        st.info('This is all the machine learning model sees when making a prediction.\n\n This is the Array converted into a GIF')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=335) 

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
    with st.expander("click for result"):
         st.write(converted_prediction)'''
st.code(code, language='python')

with st.expander("Model Outputs"):
    st.write("The code displays various outputs from the machine learning model using Streamlit.")

    st.write("1. `with col2:`")
    st.write("This line starts a context manager for rendering content inside `col2`.")
    st.write("All content within the `with col2:` block will be displayed in the right column of the app.")

    st.write("2. `st.info('This is all the machine learning model sees when making a prediction.\\n\\n This is the Array converted into a GIF')`")
    st.write("This line displays an informational message in the right column.")
    st.write("The message explains that the following visualization is what the machine learning model sees during prediction.")
    st.write("It also mentions that the array data is converted into a GIF.")

    st.write("3. `video, annotations = load_data(tf.convert_to_tensor(file_path))`")
    st.write("This line loads the video and annotations data using the `load_data` function.")
    st.write("The `video` variable will store the video frames, and the `annotations` variable will store the associated annotations.")

    st.write("4. `imageio.mimsave('animation.gif', video, fps=10)`")
    st.write("This line converts the `video` data into a GIF format and saves it as 'animation.gif'.")
    st.write("The GIF will be displayed later using `st.image`.")

    st.write("5. `st.image('animation.gif', width=335)`")
    st.write("This line displays the GIF animation in the right column using `st.image`.")
    st.write("The GIF shows the video frames that the machine learning model uses for prediction.")

    st.write("6. `st.info('This is the output of the machine learning model as tokens')`")
    st.write("This line displays an informational message indicating the following output represents tokens from the model.")

    st.write("7. `model = load_model()`")
    st.write("This line loads the machine learning model using the `load_model` function.")
    st.write("The model will be used to make predictions on the video data.")

    st.write("8. `yhat = model.predict(tf.expand_dims(video, axis=0))`")
    st.write("This line makes a prediction using the loaded model on the `video` data.")
    st.write("The `tf.expand_dims` function is used to add an extra dimension to the `video` data for batch prediction.")

    st.write("9. `decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()`")
    st.write("This line decodes the model's output using CTC decoding.")
    st.write("The `decoder` variable will store the decoded token sequences.")

    st.write("10. `st.text(decoder)`")
    st.write("This line displays the decoded token sequences as text using `st.text`.")
    st.write("The token sequences represent the model's prediction output.")

    st.write("11. `st.info('Decode the raw tokens into words')`")
    st.write("This line displays an informational message indicating the following output will be the converted prediction in words.")

    st.write("12. `converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')`")
    st.write("This line converts the decoded token sequences into words.")
    st.write("The words are stored as a string in the `converted_prediction` variable.")

    st.write("13. `with st.expander(\"click for result\"):`")
    st.write("This line starts a collapsible expander widget with the label 'click for result'.")
    st.write("The content inside this expander will be hidden initially and can be expanded by the user.")

    st.write("14. `st.write(converted_prediction)`")
    st.write("This line displays the converted prediction (words) inside the expander using `st.write`.")
    st.write("The user can click on the expander to view the result.")
    
    st.write("In summary, the code displays various outputs from the machine learning model in the right column of the app.")
    st.write("It shows the video data as a GIF animation, the model's output as tokens, and the final prediction as words.")
    st.write("The prediction is placed inside a collapsible expander for user convenience.")

st.divider()