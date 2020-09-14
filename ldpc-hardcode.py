import numpy as np
import scipy.linalg as la
import secrets

def Bitflip(bit):
    # we use this function only on specific words' bits
    # but sometimes there should be a check whether input bit or result is a part of alphabet
    return np.abs(1-bit)

def RandomizedInputWord (alphabet, N):
    result_list = []
    for i in range(N):
        bit_index = secrets.randbelow(len(alphabet))
        bit = alphabet[bit_index]
        result_list.append(bit)
    return np.array(result_list)

def BinarySymmetricalChannelWord(word, prob):
    result = []
    for bit in word:
        result.append(np.random.choice( [bit,Bitflip(bit)], p=[1-prob,prob]) )
    return np.array(result)


def CompareWordsPercent(iword, oword):
    N = iword.shape[0]
    #if iword.shape[0] != oword.shape[0]:
        # then what? change N or error?
    correctBits = 0
    for i in range(N):
        if iword[i] == oword[i]:
          correctBits += 1
    return correctBits / float(N)


# Input word (transmitter)
alphabet = [0, 1]
#iword_debug = np.array( [alphabet[1], alphabet[0], alphabet[1], alphabet[1]] )
#iword = iword_debug   # input word
iword_release = RandomizedInputWord(alphabet, 16)
iword = iword_release   # input word
print(f"Sent word:\t{iword}")


# Coder
# LDPC to be implemented
coder_word = iword  # transmitter->coder
print(f"Coded word:\t{coder_word}")


# Channel
channelBitflipProb = 0.25  # probability of a bit to flip because of channel noise
x = coder_word  # coder->channel
y = BinarySymmetricalChannelWord(x, channelBitflipProb)
print(f"Channel noise:\t{y} ({int(100-np.round(CompareWordsPercent(coder_word, y)*100))}% bitflip)")


# Decoder
# LDPC hard-decision decoder to be implemented
decoded_word = y    # channel->decoder
print(f"Decoded word:\t{y}")


# Receiver
oword = decoded_word    # decoder->receiver


# Result comparison
print(f"Received word:\t{oword}")
io_match_rate = CompareWordsPercent(iword, oword)
print(f"Success rate:\t{int(np.round(io_match_rate*100))}%")