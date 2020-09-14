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


# Input word (transmitter)
alphabet = [0, 1]
#iword_debug = np.array( [alphabet[1], alphabet[0], alphabet[1], alphabet[1]] )
#iword = iword_debug   # input word
iword_release = RandomizedInputWord(alphabet, 10)
iword = iword_release   # input word


# Coder
# LDPC to be implemented
coder_word = iword


# Channel
channelBitflipProb = 0.25  # probability of a bit to flip because of channel noise
x = coder_word
y = BinarySymmetricalChannelWord(x, channelBitflipProb)


# Decoder
# LDPC hard-decision decoder to be implemented
decoded_word = y


# Receiver
oword = decoded_word


# Result comparison
print(f"Sent word:\t{iword}")
print(f"Received word:\t{oword}")
# Correct rate to be implemented