import numpy as np

def get_scan_info(spectrum):
    scan =[spectrum['dataset'], spectrum['mzml_filename'], spectrum['scan']]
    return scan

def get_scan_info_length():
    return 3

def get_max_fragment_charge():
    return 30

def get_mono_mass_list():
    mono_mass_list = {
            "G": 57.021464,
            "A": 71.037114,
            "S": 87.032029,
            "P": 97.052764,
            "V": 99.068414,
            "T": 101.04768,
            "C": 103.00918,
            "L": 113.08406,
            "I": 113.08406,
            "D": 115.02694,
            "Q": 128.05858,
            "K": 128.09496,
            "E": 129.04259,
            "M": 131.04048,
            "m": 147.0354,
            "H": 137.05891,
            "F": 147.06441,
            "R": 156.10111,
            "Y": 163.06333,
            "N": 114.04293,
            "W": 186.07931,
            "O": 147.03538,
            "U": 150.95363,
        }
    return mono_mass_list

def get_aa_list():
    return list("ACDEFGHIKLMNOPQRSTUVWY")   

def get_char_map():
    aa_list = get_aa_list()
    aa_len = len(aa_list)
    char_map = {}
    for i, a in enumerate(aa_list):
        char_map[a] = i 
    # Special tokens "@" start, "[" end
    char_map["@"] = aa_len
    char_map["["] = aa_len + 1
    return char_map

def get_seq_encoding_dimension():
    char_map = get_char_map()
    # encode dimension + aa mass + length + position 
    return len(char_map) +  3

def get_seq_encoding_length(max_peptide_length):
    # add a start token and an end token
    return max_peptide_length + 2

def comp_proteoform_mass(proteoform):
    proteoform_mass = 0.0
    mono_mass_list = get_mono_mass_list()
    for i in range(len(proteoform)):
        aa = proteoform[i]
        if aa not in mono_mass_list:
            continue
        cur_mass = mono_mass_list.get(aa,0)
        proteoform_mass = proteoform_mass + cur_mass
    # add water mass
    proteoform_mass = proteoform_mass + 18.01056468362
    return proteoform_mass

def normalize_pos(pos, seq_len):
    norm_pos = pos / (seq_len - 1) if seq_len > 1 else pos
    norm_pos = (norm_pos * 2) - 1
    return norm_pos

def encode_spectrum(spectrum, max_peptide_length): 
    mass_scale = 200.0
    length_scale = 200.0
    mono_mass_list = get_mono_mass_list()
    char_map = get_char_map()
    char_map_size = len(char_map)
    seq_encoding_dimension = get_seq_encoding_dimension()
    seq_encoding_length = get_seq_encoding_length(max_peptide_length)
    encoding = np.zeros((seq_encoding_length, seq_encoding_dimension), dtype="float32")
    proteoform = spectrum["proteoform"]
    ori_proteoform_len = len(proteoform)
    proteoform = "@" + proteoform + "["
    for i, aa in enumerate(proteoform):
        if aa not in char_map: 
            continue
        # one hot coding 
        encoding[i][char_map[aa]] = 1                                          
        if i > 0 and i < ori_proteoform_len+1:
            # encode amino acid mass
            encoding[i][char_map_size] = mono_mass_list.get(aa,0) / mass_scale 
            # encode proteoform length
            encoding[i][char_map_size + 1] = ori_proteoform_len / length_scale        
            # position info
            encoding[i][char_map_size + 2] = normalize_pos(i, ori_proteoform_len)     
    return encoding

def get_meta_length():
    return 46

def get_activation_map():
    activation_types = {"unknown":0,"cid":1,"etd":2,"hcd":3,"ethcd":4}
    return activation_types

# meta dimension after embedding
def get_meta_embedding_dimension():
    return 8

def encode_meta(spectrum):
    # charge
    max_charge = get_max_fragment_charge()
    charge_encoding = np.zeros(max_charge,dtype="float32")
    charge = spectrum["prec_charge"]
    if charge >=1 and charge <=max_charge:
        charge_encoding[charge - 1] = 1  
    else:
        print("Warning: Unknown precursor charge:", charge)

    # instrument
    instrument_encoding = np.zeros(9,dtype="float32")
    instrument_map = {"unknown":0,"ltq ft ultra":1,"ltq orbitrap elite":2,"orbitrap eclipse":3,"orbitrap fusion lumos":4,
                      "q exactive":5, "q exactive hf":6, "q exactive plus":7, "orbitrap exploris 240":8}
    instrument = spectrum["instrument"].lower()
    instrument_code = instrument_map.get(instrument, 0)
    if instrument_code == 0:
        print("Warning: Unknown instrument:", instrument)
    # activation
    activation_encoding = np.zeros(5,dtype="float32")
    activation_map = get_activation_map()
    activation = spectrum["activation_type"]
    activation_code = activation_map.get(activation, 0)
    if activation_code == 0:
        print("Warning: Unknown activation type:", activation)
    activation_encoding[activation_code] = 1 

    # precursor mass and nce
    precursor_scale = 10000.0
    norm_proteoform_mass = comp_proteoform_mass(spectrum["proteoform"]) / precursor_scale
    nce = 0.25
    if not "nce" in spectrum or spectrum["nce"] == 0:
        nce = 0.25
    else:
        nce = spectrum["nce"] / 100.0
    meta_length = get_meta_length()
    meta = np.zeros((meta_length), dtype="float32")
    meta[0:30] = charge_encoding
    meta[30:39] = instrument_encoding
    meta[39:44] = activation_encoding
    meta[44] = norm_proteoform_mass
    meta[45] = nce
    #print("meta shape:", meta.shape)
    return meta

def get_mask(spectrum, max_seq_length):
    seq_encoding_length = get_seq_encoding_length(max_seq_length)
    mask = np.zeros(seq_encoding_length, dtype="float32")      
    # Mask for proteoform sequence only. Do not cover start and end tokens  
    mask[1:len(spectrum["proteoform"])+1] = 1.0
    return mask

def get_charge_mask(prec_charge):
    max_charge = get_max_fragment_charge()
    charge_mask = np.zeros((max_charge*2), dtype="float32")
    if prec_charge < 1 or prec_charge > max_charge:
        print("Warning: Unknown precursor charge for charge mask:", prec_charge)
        return charge_mask
    # b ions
    charge_mask[0: prec_charge] = 1.0
    # y ions
    charge_mask[max_charge: max_charge + prec_charge] = 1.0
    return charge_mask  