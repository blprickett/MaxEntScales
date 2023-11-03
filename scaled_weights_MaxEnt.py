import numpy as np
from sys import exit
from scipy.optimize import minimize
from datetime import datetime
from sys import argv
import re
from math import ceil
#This is a modified version of the scaled-weights
#MaxEnt model described in Hughto et al. (2019).

#####USER SETTINGS#####   
#General parameters:
METHOD = argv[1]#Choose from the set {GD, GD_CLIP, and L-BFGS-B}
NEG_WEIGHTS = bool(int(argv[2])) #Set this to 1 if you're cool with negative weights
NEG_SCALES = bool(int(argv[3])) #Set this to 1 if you're cool with negative scales

priors = argv[4].split(",")
if len(priors) == 2:
    LAMBDA1 = float(priors[0])#Weight for regularization of general weights
    LAMBDA2 = float(priors[1])#Weight for regularization of scales
elif len(priors) == 1:
    LAMBDA1 = float(priors[0]) #Weight for regularization of general weights
    LAMBDA2 = LAMBDA1 #Weight for regularization of scales
else:
    raise Exception("Priors argument must have one value or two comma-delimited values!")

L2_PRIOR = bool(int(argv[5]))   #L2 prior? (L1 is alternative) 
if (not L2_PRIOR) and (METHOD == "GD"):
    raise Exception("Must use an L2 prior with vanilla gradient descent! Please use the 'GD_CLIP' method if you want to use an L1 prior!")
if (L2_PRIOR) and (METHOD == "GD_CLIP"):
    raise Exception("Must use an L1 prior with clipped gradient descent! Please use the vanilla 'GD' method if you want to use an L2 prior!")   

init_weights = argv[6].split(",")
if len(init_weights) == 1:
    INIT_WEIGHT_C = float(init_weights[0]) #Initial weights for constraints (if RAND_WEIGHTS == 0)
    INIT_WEIGHT_S = float(init_weights[0]) #Initial weights for scales (if RAND_WEIGHTS == 0)
elif len(init_weights) == 2:
    INIT_WEIGHT_C = float(init_weights[0]) #Initial weights for constraints (if RAND_WEIGHTS == 0)
    INIT_WEIGHT_S = float(init_weights[1]) #Initial weights for scales (if RAND_WEIGHTS == 0)
else:
    raise Exception("Weights argument must have one value or two comma-delimited values!")

RAND_WEIGHTS = bool(int(argv[7])) #Makes initial weights randoms ints between 0-10

#Params for gradient descent:
rates = argv[8].split(",")
if len(rates) == 1:
    ETA1 = float(rates[0])     #Learning rate for constraints
    ETA2 = float(rates[0])     #Learning rate for scales
elif len(rates) == 2:
    ETA1 = float(rates[0])     #Learning rate for constraints
    ETA2 = float(rates[1])     #Learning rate for scales
else:
    raise Exception("Learning rates argument must have one value or two comma-delimited values!")
EPOCHS = int(argv[9])  #Iterations through whole dataset

#Output file settings
LANGUAGE = argv[10] #Arbitrary label used in the input/output files

#####FUNCTIONS##### 
def get_predicted_probs (weights, scales, viols):
    #First we need to add up the scales for the morphemes in each datum:
    scales_by_datum = np.array([np.sum(scales[datum2morphs[datum]], axis=0)\
                                        for datum in range(len(p))])
    #Then we add the scales to the weights:
    scaledWeights_by_datum = scales_by_datum + weights 
    
    #Simple MaxEnt stuff:
    harmonies = np.sum(viols * scaledWeights_by_datum, axis=1)
    eharmonies = np.exp(-1 * harmonies)
    Z_by_UR = np.array([sum(eharmonies[ur2data[ur_tokens[datum]]]) \
                            for datum, viol in enumerate(viols)])
    probs = eharmonies/Z_by_UR

    return probs
    
def get_nonce_probs (weights, viols):
    harmonies = viols.dot(weights)
    eharmonies = np.exp(-1 * harmonies)
    Z_by_UR = np.array([sum(eharmonies[ur2data[ur_tokens[datum]]]) \
                        for datum, viol in enumerate(viols)])
    probs = eharmonies/Z_by_UR
    
    return probs    

def grad_descent_update (weights, viols, td_probs, scales, eta1, eta2):
    #Dimensions (repeated for new variables throughout):
    #Weights -> C (where C is the # of constraints)
    #td_probs -> D (where D is the # of data)
    #Viols -> DxC
    #Scales -> MxC (where M is the # of morphemes)
    
    #Forward pass (learner's expected probabilities):
    le_probs = get_predicted_probs (weights, scales, viols) #(D)
    
    #Backward pass:
    TD_byDatum = viols.T * td_probs #Violations by datum present in the training data (CxD)
    LE_byDatum = viols.T * le_probs #Violations by datum expected by the learner (CxD)
    
    #Convert the expected violations by datum to expected violations by morpheme 
    #(this could probably be more efficient):
    TD_byMorph = np.zeros(scales.shape) #Violations by morph present in the training data (MxC)
    LE_byMorph = np.zeros(scales.shape) #Violations by morph expected by the learner (MxC)
    for datum_index, datum2morph in enumerate(datum2morphs):
        for morph in datum2morph:
            TD_byMorph[morph] += TD_byDatum.T[datum_index]
            LE_byMorph[morph] += LE_byDatum.T[datum_index] 
    
    #The part of the gradients from the log loss is TD-LE (obs. - exp.)
    c_gradients = np.sum(TD_byDatum, axis=1) - np.sum(LE_byDatum, axis=1) #(C)
    s_gradients = TD_byMorph - LE_byMorph #(MxC)
    
    #Update based on log loss:
    almost_new_weights = weights - (c_gradients * eta1)#aka w^{k+1/2} (C)
    almost_new_scales = scales - (s_gradients * eta2)#(MxC)
    
    #Updates based on prior:
    if "CLIP" in METHOD:
        #With clipping (Tsuruoka et al. 2009):
        new_weights = []
        for w_kPlusHalf in almost_new_weights:
            if w_kPlusHalf > 0:
                #If the weight is above zero, 
                #don't let the prior take it below zero.
                w_kPlus1 = max([0, w_kPlusHalf - ((LAMBDA1)*eta1)])
            elif w_kPlusHalf < 0:
                #If the weight is below zero, don't let the prior take it
                #above zero.
                w_kPlus1 = min([0, w_kPlusHalf + ((LAMBDA1)*eta1)])
            else:
                #If the weight is exactly zero, don't change it.
                w_kPlus1 = w_kPlusHalf
            new_weights.append(w_kPlus1)
        new_scales = []
        for morpheme_scales in almost_new_scales:
            this_morphs_scales = []
            for s_kPlusHalf in morpheme_scales:
                if s_kPlusHalf > 0:
                    #If the scale is above zero, 
                    #don't let the prior take it below zero.
                    s_kPlus1 = max([0, s_kPlusHalf - ((LAMBDA2)*eta2)])
                elif s_kPlusHalf < 0:
                    #If the scale is below zero, don't let the prior take it
                    #above zero.
                    s_kPlus1 = min([0, s_kPlusHalf + ((LAMBDA2)*eta2)])
                else:
                    #If the scale is exactly zero, don't change it.
                    s_kPlus1 = s_kPlusHalf
                this_morphs_scales.append(s_kPlus1)
            new_scales.append(this_morphs_scales)
            
        new_weights = np.array(new_weights)
        new_scales = np.array(new_scales)
    else:
        #Without clipping:
        prior1_gradient = (LAMBDA1 * weights) #(C)
        prior2_gradient = (LAMBDA2 * scales) #(MxC)
        
        new_weights =  almost_new_weights - prior1_gradient * eta1 #aka w^{k+1}
        new_scales = almost_new_scales - prior2_gradient * eta2
    
    #And only have negative stuff if we want to allow it:
    if not NEG_WEIGHTS:
        new_weights = np.maximum(new_weights, 0)
    if not NEG_SCALES:
        new_scales = np.maximum(new_scales, 0)
        
    return new_weights, new_scales

def objective_function (weightsAndScales, viols, td_probs, areScalesFlat=True):
    #This function is used by the LBFGSB optimizer:
    to_return = ()

    #Handle "weightsAndScales" differently, depending on the areScalesFlat parameter:
    if areScalesFlat:
        weights, scales = weightsAndScales[:len(w)], weightsAndScales[len(w):]
        scales = np.reshape(scales, s.shape)
    else:
        weights = weightsAndScales[0]
        scales = weightsAndScales[1:]
        
    #Regular log loss stuff:
    le_probs = get_predicted_probs(weights, scales, viols)
    log_probs = np.log(le_probs)
    loss = (-1.0 * np.sum(td_probs*log_probs)) #log loss
    to_return += (loss,) #keep track of loss separate from pior

    if L2_PRIOR:
        loss += (np.sum(weights**2)*LAMBDA1) + (np.sum(scales**2)*LAMBDA2) #L2 regularization
        to_return += (None,)
        to_return += (loss,)
    else:
        prior = (np.sum(np.abs(weights))*LAMBDA1) + (np.sum(np.abs(scales))*LAMBDA2) #L1 regularization
        to_return += (prior,)#keep track of prior separately
        loss += prior
        to_return += (loss,)#full objective fnc.
    if "GD" in METHOD:
        return to_return
    else:
        return to_return[-1]
    
#####PROCESS LEARNING DATA##### 
tableaux_file = open("Training_Data/"+LANGUAGE+"_grammar.txt", "r") #file w/ violation profiles and candidates
distribution_file = open("Training_Data/"+LANGUAGE+"_dist.txt", "r") #file w/ UR->morph mappings
start_time = re.sub(":", ",", str(datetime.now())) #time stamp for output files

#Process dist file:
morphemes = []
mapping2freq = {}
UR2morphs = {}
UR2totalFreq = {}
for dist_line in distribution_file.readlines():
    mapping_match = re.search('"(.+)"\s+"(.+)"\s+([0-9.]+)\s+([0-9,]+)\s+(.+)', dist_line)
    if not mapping_match:
        continue
    this_UR = mapping_match.group(1)
    this_SR = mapping_match.group(2)
    this_freq = float(mapping_match.group(3))
    this_morphList = mapping_match.group(4).split(",")
    
    if this_freq % 1 != 0:
        print("**Warning! All non-integer frequencies will be rounded up to the nearest integer!")
        this_freq = float(ceil(this_freq))
    
    morphemes += this_morphList
    mapping2freq[this_UR+"->"+this_SR] = this_freq
    UR2morphs[this_UR] = this_morphList
    try:
        UR2totalFreq[this_UR] += this_freq
    except:
        UR2totalFreq[this_UR] = this_freq
morphemes = list(set(morphemes))

#Process grammar file:
c_names = [] #List of the constraint names
URs = [] #A list the UR for each datum (i.e. each candidate)
raw_SRs = [] #A list of every SR
raw_v = [] #This will store each candidate's violation profile
raw_p = [] #This will store each candidate's conditional probability, i.e. Pr(SR|UR)
raw_datum2morphs = [] #A list of the lists of the morphs for each datum
for gram_line in tableaux_file.readlines():
    gram_line.rstrip()
    cName_match = re.search('constraint \[.+\]: "(.+)"', gram_line)
    UR_match = re.search('input \[.+\]: "(.+)"', gram_line)
    SR_match = re.search('candidate \[.+\]: "(.+)" ([0-9 ]+)', gram_line)
    if cName_match:
        cName = cName_match.group(1)
        c_names.append(cName)
        continue
    if UR_match:
        this_UR = UR_match.group(1)
        continue
    if SR_match:
        this_SR = SR_match.group(1)
        these_viols = SR_match.group(2).split(" ")
        URs.append(this_UR)
        raw_SRs.append(this_SR)
        raw_v.append([float(tv) for tv in these_viols])
        if this_UR+"->"+this_SR in mapping2freq.keys():
            raw_p.append(float(mapping2freq[this_UR+"->"+this_SR])/float(UR2totalFreq[this_UR]))
        else:
            raw_p.append(0.0)
        raw_datum2morphs.append([morphemes.index(m) for m in UR2morphs[this_UR]])

#Create a dictionary for efficiently finding Z's:
#(Also factor in token frequency here...)
ur2data = {}
ur_tokens = []
v = []
p = []
SRs = []
datum2morphs = []
datum_index = 0
for ur_type, ur in enumerate(URs):
    for ur_token in range(ceil(UR2totalFreq[ur])):
        v.append(raw_v[ur_type])
        p.append(raw_p[ur_type])
        datum2morphs.append(raw_datum2morphs[ur_type][:])
        SRs.append(raw_SRs[ur_type])
        
        token_id = ur+"_"+str(ur_token)
        if token_id in ur2data.keys():
            ur2data[token_id].append(datum_index)
        else:
            ur2data[token_id] = [datum_index]
        ur_tokens.append(token_id)
        datum_index += 1  

i = -1
for i, sr in enumerate(SRs):
    if sr[-2:] == "ka":
        gen_index = i
        break
        
if i == -1:
    raise Exception("Couldn't find any SR's ending in 'ka'!")        

#All the arrays we need: 
v = np.array(v)  #Constraint violations
if RAND_WEIGHTS:
    w = np.random.uniform(low=0.0, high=10.0, size=len(v[0]))
    s = np.array([np.random.uniform(low=0.0, high=10.0, size=len(v[0])) for morph in morphemes])
    weights_file = open("Raw_Output/"+start_time+"_"+LANGUAGE+"_initWeights.txt", "w")
    for c, name in enumerate(c_names):
        weights_file.write(name+"\n\t"+str(w[c])+"\n")
        for this_s in s[c]:
            weights_file.write("\t"+str(s[c])+"\n")
    weights_file.close()
else:
    w = np.array([INIT_WEIGHT_C for c in v[0]])     
    s = np.array([[INIT_WEIGHT_S for c in c_names] for morph in morphemes]) #Initial scales    
p = np.array(p) #Training data probs
all_params = np.concatenate((w,np.ndarray.flatten(s)))

#####LEARNING##### 
if "GD" in METHOD:
    init_loss = objective_function(all_params, v, p)[0]
    loss_tracker = []
    tdprob_tracker = [{}]
    gen_tracker = []
    min_loss = 1000000
    best_weights = []
    best_scales = []
    best_epoch = -1
    for ep in range(EPOCHS):
        #Concatenate scales and weights:
        full_params = np.concatenate((np.array([w]), s))
        
        #Find and save loss, given the current params:
        this_loss = objective_function(full_params, v, p, False)
        loss_tracker.append(this_loss)
        
        #Find and save training data probs, given the current params:
        my_td_probs = get_predicted_probs(w, s, v)
        tdprob_tracker.append(my_td_probs)
        
        #Find and save novel data probs, given the current params:
        this_gen = get_nonce_probs(w, v)[gen_index]
        gen_tracker.append(this_gen)
        
        #Save the best epoch in learning:
        if this_loss[-1] < min_loss:
            best_weights = w
            best_scales = s
            best_epoch = ep
            min_loss = this_loss[-1]
        if ep % 1000 == 0:
            print ("Epoch: "+str(ep)+"\tLoss, Prior, Obj: "+str(this_loss))
            print ("\tweights: "+str(w))
            
        w, s = grad_descent_update(w, v, p, s, eta1=ETA1, eta2=ETA2)
    final_weights = w
    final_params = np.concatenate((np.array([final_weights]), s))
    final_scales = s
    
    print ("Beginning loss: ", init_loss)
    print  ("Final loss: ", objective_function(final_params, v, p, False))
else:
    init_loss = objective_function(all_params, v, p)
    if NEG_WEIGHTS:
        min_w = None
    else:
        min_w = 0.0
    final_params = minimize(objective_function, all_params, args=(v, p), method=METHOD, bounds=[(min_w, None) for x in all_params])['x']
    final_weights = final_params[:len(w)]
    final_scales = np.reshape(final_params[len(w):], s.shape)
    
    print ("Beginning loss: ", init_loss)
    print ("Final loss: ", objective_function(final_params, v, p))

#####OUTPUT#####
if RAND_WEIGHTS:
    iw_desc = "Random"
else:
    iw_desc = str(INIT_WEIGHT_C)+","+str(INIT_WEIGHT_S)
    

#Print final state of the grammar:
print ("Printing grammar to file...")
WEIGHT_file = open("Raw_Output/"+start_time+"_"+LANGUAGE+"_outputWeights_"+"C="+str(argv[4])+"_lr="+str(argv[8])+"_iw="+iw_desc+".csv", "w")
WEIGHT_file.write("Morpheme,"+",".join(c_names)+"\n")
WEIGHT_file.write("General,"+",".join([str(fw) for fw in final_weights])+"\n")

for m_index, this_morph in enumerate(morphemes):
    m_weights = final_scales[m_index]
    WEIGHT_file.write(this_morph+","+",".join([str(mw) for mw in m_weights])+"\n")
WEIGHT_file.close()

#Print nonce judgments:
print ("Printing nonce judgments to file...")
NONCE_file = open("Raw_Output/"+start_time+"_"+LANGUAGE+"_nonceProbs_"+"C="+str(argv[4])+"_lr="+str(argv[8])+"_iw="+iw_desc+".csv", "w")
NONCE_file.write("Input,Output,ObservedProb,ExpectedProb\n")
my_nonce_probs = get_nonce_probs(final_weights, v)
for datum_index, datum_prob in enumerate(my_nonce_probs):
    ur_string = ur_tokens[datum_index].split("$")[0]
    NONCE_file.write(ur_string+","+SRs[datum_index]+","+\
                        str(p[datum_index])+","+str(datum_prob)+"\n")   
NONCE_file.close()

#Print training data judgments:
print ("Printing training data judgments to file...")
TDJ_file = open("Raw_Output/"+start_time+"_"+LANGUAGE+"_tdProbs_"+"C="+str(argv[4])+"_lr="+str(argv[8])+"_iw="+iw_desc+".csv", "w")
TDJ_file.write("Input,Output,ObservedProb,ExpectedProb\n")
my_td_probs = get_predicted_probs(final_weights, final_scales, v)
for datum_index, datum_prob in enumerate(my_td_probs):
    ur_string = ur_tokens[datum_index].split("$")[0]
    TDJ_file.write(ur_string+","+SRs[datum_index]+","+\
                        str(p[datum_index])+","+str(datum_prob)+"\n")   
TDJ_file.close()

if "GD" in METHOD:
    #Print info for the epoch with the lowest loss:
    print ("Printing best epoch data to file...")
    bestEP_file = open("Raw_Output/"+start_time+"_"+LANGUAGE+"_bestEpoch_"+"C="+str(argv[4])+"_lr="+str(argv[8])+"_iw="+iw_desc+".txt", "w")
    bestEP_file.write("Best epoch was "+str(best_epoch)+"\n")
    bestEP_file.write("Loss at best epoch was "+str(min_loss)+"\n")
    bestEP_file.write("Weights:\n")
    for c_i, bw in enumerate(best_weights):
        bestEP_file.write(c_names[c_i]+"\t"+str(bw)+"\n")
    bestEP_file.write("Scales:\n")
    for m_i, bs in enumerate(best_scales):
        bestEP_file.write(morphemes[m_i]+"\n")
        for c_i, this_s in enumerate(bs):
            bestEP_file.write("\t"+c_names[c_i]+"\t"+str(this_s)+"\n") 
    bestEP_file.close()
    
    
    #Print learning curve:
    print("Printing learning curve data...")
    CURVE_file = open("Raw_Output/"+start_time+"_"+LANGUAGE+"_learningCurve_"+"C="+str(argv[4])+"_lr="+str(argv[8])+"_iw="+iw_desc+".csv", "w")
    CURVE_file.write("Epoch,LogLoss,Prior,FullLoss\n")
    for ep, loss in enumerate(loss_tracker):
        ll, my_p, fl = loss
        CURVE_file.write(",".join([str(ep), str(ll), str(my_p), str(fl)])+"\n")
        
    #Print training data probabilities curve:
    TDP_CURVE_file = open("Raw_Output/"+start_time+"_"+LANGUAGE+"_tdProbCurve_"+"C="+str(argv[4])+"_lr="+str(argv[8])+"_iw="+iw_desc+".csv", "w")
    TDP_CURVE_file.write("Input,Output,Epoch,ObservedProb,ExpectedProb\n")
    for ep, TDPs in enumerate(tdprob_tracker):
        for datum_index, datum_prob in enumerate(TDPs):
            ur_string = ur_tokens[datum_index].split("$")[0]
            TDP_CURVE_file.write(ur_string+","+SRs[datum_index]+","+str(ep)+","+\
                                    str(p[datum_index])+","+str(datum_prob)+"\n")   
    TDP_CURVE_file.close()
        
    #Print generalization curve:
    print("Printing generalization curve data...")
    gen_file = open("Raw_Output/"+start_time+"_"+LANGUAGE+"_generalizationCurve_"+"C="+str(argv[4])+"_lr="+str(argv[8])+"_iw="+iw_desc+".csv", "w")
    gen_file.write("Epoch,ProbOfMajSuff\n")
    for ep, g in enumerate(gen_tracker):
        gen_file.write(str(ep)+","+str(g)+"\n")
    gen_file.close()

#All done!
print ("All done!" )