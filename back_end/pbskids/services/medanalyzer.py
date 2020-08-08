import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer
import torch
from transformers import BertForQuestionAnswering
import pandas as pd
import re
#import openpyxl
from googletrans import Translator
import speech_recognition as sr
import os
from django.conf import settings
from moviepy.editor import *
from google.cloud import speech_v1
from google.cloud.speech_v1 import enums
import io
import soundfile as sf
from pydub import AudioSegment
import numpy as np
from django.core.files.storage import FileSystemStorage
import gc


def unloadModule(mod):    
	# removes module from the system    
	mod_name = mod.__name__    
	if mod_name in sys.modules:        
		del sys.modules[mod_name]

def answer_question_part(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Used for parts of  large text. 
    Returns the best answer and average of the max start and end scores.
    '''
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')      
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)
    # Report how long the input sequence is. This can be commented out if desired.
    print('Query has {:,} tokens.\n'.format(len(input_ids)))
    
      
    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1
    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    
    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    del tokenizer
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    
    # ======== Evaluate ========
    # Run our example question through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    del model   
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad') 
    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Pull the scores out of PyTorch Tensors and convert them to 1D numpy arrays.
    s_scores = start_scores.detach().numpy().flatten()
    e_scores = end_scores.detach().numpy().flatten()

    # We'll use the tokens as the x-axis labels. In order to do that, they all need
    # to be unique, so we'll add the token index to the end of each one.
    token_labels = []
    for (i, token) in enumerate(tokens):
        token_labels.append('{:} - {:>2}'.format(token, i))

    # Start with the first token.
    answer = tokens[answer_start]
    
    #Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]
    gc.collect()
    # Returns answer and average of max start and end scores.
    return answer,  (max(s_scores) + max(e_scores))/2


def longtext_answer(question, answer_text):
    '''
    To be used for longer text, using the answer_question_part function 
    (number of tokens being printed out can be removed from that if desired)
  '''
  # Average length of word, or length of slice of text, it might seem like it should be longer, but 4 give token numbers close to the max
    avg_word_len =1
    # Max number of tokens allowed (it's always 512, for now)
    max_tokens = 512
    # Assume that the string won't have more than the average word length multiplied by the number of tokens
    increment = int(avg_word_len*max_tokens)-len(question)
    # List for scores for each answer for each slice of the string
    scores = []
    # List of answers for each slice of the string
    answers = []
    
    
    chunks=1 if int((len(answer_text))/increment)==0 else int(len(answer_text)/increment)
    print(chunks)
    # Loop through the text string, in increments chosen above
    for i in range(chunks):
        # Checks for best answer starting from 0 in the string and appends answers and score values to lists
        ans, score = answer_question_part(question, answer_text[0+i*increment:increment+i*increment])
        answers.append(ans)
        scores.append(score)
        # Checks for best answer starting from half of the increment, in case the best answer is split between increments, and appends values
        ans, score = answer_question_part(question, answer_text[int(increment/2+i*increment):int(increment+increment/2+i*increment)])
        answers.append(ans)
        scores.append(score)
    # Gets the index for the best score  
    max_score_ind = np.array(scores).argmax()
    # Gives the answer that matches the best score index
    best_answer = answers[max_score_ind]
    # Returns the best answer
    return best_answer

def frame_rate_channel(audio_file_name):
    f = sf.SoundFile(audio_file_name)
    frame_rate = f.samplerate
    channels = f.channels
    return frame_rate,channels

def convert_to_audio():
    
    for file in os.listdir(settings.DATA_ROOT):
                 fl_path=os.path.join(settings.DATA_ROOT, file)
            

    audfile='convo.wav'
    fl = open(fl_path, 'rb')

    audioclip = AudioFileClip(fl_path) #colab doesn't save files so this would have to be downloaded and uploaded from slack (or linked to if permissions work)
    audioclip.write_audiofile(os.path.join(settings.IMG_ROOT, audfile))
    

def extract_med_info():
  
    convert_to_audio()
 
    audfile='convo.wav'
    
    fl_path=os.path.join(settings.BASE_DIR, 'Genv.json')
    sample_rate_hertz, channelcount= frame_rate_channel(os.path.join(settings.IMG_ROOT, audfile))
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = fl_path
    client = speech_v1.SpeechClient()

    # The language of the supplied audio
    language_code = "en-US"

    # Encoding of audio data sent. This sample sets this explicitly.
    # This field is optional for FLAC and WAV audio formats.
    encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
    config = {
        "language_code": language_code,
        "sample_rate_hertz": sample_rate_hertz,
        "encoding": encoding,
        "audio_channel_count": channelcount,
    }
    
    convotxt=""
    
    newAudio = AudioSegment.from_file(os.path.join(settings.IMG_ROOT, audfile))
    for t in range(0, int(newAudio.duration_seconds)//30):
        tmp = newAudio[t*30000:(t+1)*30000]
        tmp.export(os.path.join(settings.IMG_ROOT, 'chunk.wav'), format="wav")   


        with io.open(os.path.join(settings.IMG_ROOT, 'chunk.wav'), "rb") as f:
            content = f.read()
        audio = {"content": content}

        operation = client.long_running_recognize(config, audio)

        print(u"Waiting for operation to complete...")
        response = operation.result()

        for result in response.results:
            # First alternative is the most probable result
            alternative = result.alternatives[0]
            convotxt+=alternative.transcript
            
    fs = FileSystemStorage()
    fs.delete(os.path.join(settings.IMG_ROOT, 'chunk.wav'))
            
    translator = Translator()

    answertxt=""
    answerfile=os.path.join(settings.RESULT_ROOT, 'answer.txt')
    
    
    answertxt=longtext_answer("when was the last diagnosis?", convotxt)
    
    file = fs.open(answerfile, "w+") 
    file.write("when was the last diagnosis?: "+answertxt+"\n") 
    
    answertxt=longtext_answer("what medication do you take?", convotxt)
    
    file = fs.open(answerfile, "a+") 
    file.write("what medication do you take?: "+answertxt+"\n") 
            
    
    
    return answerfile
