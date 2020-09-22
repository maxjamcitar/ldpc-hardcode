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
    result[0:n_ones] = 1.0
    np.random.shuffle(result)
    return result


def BitFlippingLDPCDecoding(word, H):
    result_word = np.copy(word)
    isDecodingOver = False
    max_iterations = 100
    for i_iter in range(max_iterations):
        parityFailsRows = []    # indices of H rows with failed parity check
        for rowIndex, row in enumerate(H):
            bdp = BinaryDotProduct(row, result_word)
            if bdp != 0:
                parityFailsRows.append(rowIndex)
        if len(parityFailsRows) == 0:
            break   # all parity checks are satisfied
        codewordParityFails = np.zeros_like(word)
        for rowIndex in parityFailsRows:
            ones = np.where(H[rowIndex] == 1)[0] # Tanner graph lines (downwards)
            for oneIndex in ones:
                codewordParityFails[oneIndex] += 1
        bitFlipIndex = np.where(codewordParityFails == np.amax(codewordParityFails))[0][0]  # first max
        result_word[bitFlipIndex] = Bitflip(result_word[bitFlipIndex])
    return result_word


def BinaryDotProduct(row, word):
    if row.ndim != 1 or word.ndim != 1:
        raise(Exception("Multidimensional array in a vector dot product function"))
    alphabetModule = 2
    return np.dot(row, word) % alphabetModule


channelBitflipProb = 0.1  # probability of a bit to flip because of channel noise

# Input word (transmitter)
alphabet = [0.0, 1.0]
#iword_debug = np.array( [alphabet[1], alphabet[0], alphabet[1], alphabet[1]] )
#iword = iword_debug   # input word
iword_release = RandomizedInputWord(alphabet, 10)
iword = iword_release   # input word
print(f"Sent word:\t{iword}")

n0 = 5  # H: ones per row
l = 3   # H: one-stacks 
m = 3   # H: rows in a block
bl = 3  # H: H0 blocks
k_bits = iword.shape[0] # G: number of information bits
n_cwl = n0*l    # G: CodeWord Length
# G = ? How...
H = LDPCCheckingMatrix(n0, l, m, bl)


# Encoder
word_encoded = iword  # transmitter->encoder
# todo: proper encoding
encodedWordPiece = np.zeros(n_cwl - k_bits)
word_encoded = np.append(word_encoded, encodedWordPiece)    # placeholder codeword
print(f"Encoded word:\t{word_encoded}")


# Channel
x_chan = word_encoded  # encoder->channel
y_chan = BinarySymmetricalChannelWord(x_chan, channelBitflipProb)   # noisy channel
print(f"Channel noise:\t{y_chan} ({int(100-np.round(CompareWordsPercent(word_encoded, y_chan)*100))}% bitflip)")


# Decoder
word_decoded = y_chan    # channel->decoder
word_decoded = BitFlippingLDPCDecoding(word_decoded, H)
print(f"Decoded word:\t{word_decoded}")


# Receiver
# do we know k_bits at this point?
oword = word_decoded[0:k_bits]    # decoder->receiver


# Result comparison
print(f"Received word:\t{oword}")
io_match_rate = CompareWordsPercent(iword, oword)
print(f"Success rate:\t{int(np.round(io_match_rate*100))}%")