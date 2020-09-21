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


def LDPCCheckingMatrix(n0, l, m, bl):
    rows_H = m*bl
    cols_H = n0*l
    H = np.zeros(shape=(rows_H, cols_H))
    # H0: first m rows
    for i_row in range(m):
        H[i_row][i_row*n0:(i_row+1)*n0] = 1.0
    # the rest bl-1 blocks (random ones placement)
    for i_block in np.arange(1, bl, 1):
        for i_row in np.arange(i_block*m, (i_block+1)*m, 1):
            H[i_row] = RandomizedSparseOnesVector(cols_H, n0)
    return H


# binary vector with randomly placed n_ones ones ('pi')
def RandomizedSparseOnesVector(size, n_ones):
    result = np.zeros(size)
    result[0:n_ones] = 1
    np.random.shuffle(result)
    return result



# Input word (transmitter)
alphabet = [0, 1]
#iword_debug = np.array( [alphabet[1], alphabet[0], alphabet[1], alphabet[1]] )
#iword = iword_debug   # input word
iword_release = RandomizedInputWord(alphabet, 16)
iword = iword_release   # input word
print(f"Sent word:\t{iword}")

k_bits = iword.shape[0] # G: number of information bits
n_cwl = k_bits + 100    # G: CodeWord Length
n0 = 5  # H: ones per row
l = 3   # H: one-stacks 
m = 3   # H: rows in a block
bl = 3  # H: H0 blocks
# G = ? How...
H = LDPCCheckingMatrix(n0, l, m, bl)


# Encoder
word_encoder = iword  # transmitter->encoder
# I actually need to encode it. But more on that later...
print(f"Encoded word:\t{word_encoder}")


# Channel
channelBitflipProb = 0.1  # probability of a bit to flip because of channel noise
x_chan = word_encoder  # encoder->channel
y_chan = BinarySymmetricalChannelWord(x_chan, channelBitflipProb)
print(f"Channel noise:\t{y_chan} ({int(100-np.round(CompareWordsPercent(word_encoder, y_chan)*100))}% bitflip)")


# Decoder
print(H)
# c*H.T = 0 check
# majority decoding thing
word_decoded = y_chan    # channel->decoder
print(f"Decoded word:\t{word_decoded}")


# Receiver
oword = word_decoded    # decoder->receiver


# Result comparison
print(f"Received word:\t{oword}")
io_match_rate = CompareWordsPercent(iword, oword)
print(f"Success rate:\t{int(np.round(io_match_rate*100))}%")