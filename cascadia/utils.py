from depthcharge.tokenizers.peptides import PeptideTokenizer, Peptide,  _calc_precursor_mass, nb
import numpy as np
import re

def write_results(preds, results_file, raw_file, isolation_window_size, score_threshold, time_width):
    pred_seqs, aa_conf, pep_conf, rts, precursors = [], [], [], [], []

    for pred_seqs_c, _, pep_conf_c, aa_conf_c, rt_c, precursors_c in preds:
        pred_seqs.append(pred_seqs_c)
        aa_conf.append(aa_conf_c)
        pep_conf.append(pep_conf_c)
        rts.append(rt_c)
        precursors.append(precursors_c)
        
    pred_seqs = np.concatenate(pred_seqs)
    pep_conf = np.concatenate(pep_conf)
    rts = np.concatenate(rts)
    precursors = np.concatenate(precursors)

    if max(rts) > 500:
      rts = rts/60
      time_width = time_width/60

    tokenizer = PeptideTokenizer.from_massivekb(reverse=False, replace_isoleucine_with_leucine=True)
    with open(results_file + '.ssl', 'w') as out:
        out.write("\t".join(['file',	'scan',	'charge',	'sequence',	'score-type',	'score',	'retention-time',	'start-time',	'end-time']) + '\n')
        for idx, (pred, conf, rt, prec) in enumerate(zip(pred_seqs, pep_conf, rts, precursors)):
            conf = np.exp(conf)
            unmodified = re.sub("\[.*?\]","", pred)
            if '-' in pred:
                if unmodified.rfind('-') != 0:
                    continue
            if len(unmodified) > 0 and conf > score_threshold:
              mz = prec[0]
              charge = prec[1]
              mz = mz / charge + 1.007276
              mz = mz / charge + 1.007276
              pred_mz = _calc_precursor_mass(nb.typed.List(Peptide.from_massivekb(pred).split()), charge, tokenizer.residues)
              if np.abs(pred_mz - mz) < 2*isolation_window_size:
                out.write("\t".join([raw_file, str(idx), str(charge), pred, 'Cascadia Score', str(conf), str(rt), str(rt-time_width), str(rt+time_width)]) + '\n')