
import json
from scripts import write
import os
from formatlatex.semester_10_relation_property_hypothesis import FormatRelationPropertyHypothesis
from formatlatex.semester_10_voting_hypothesis import FormatVotingHypothesis
from formatlatex.semester_10_time_density import FormatTimeDensity
from formatlatex.semester_10_error_distribution import FormatErrorDistribution
from formatlatex.texobject import texobject

class FormatLatex():
    def __init__(self, params, task = []) -> None:
        self.params = params
        self.task = task

    def sort_methods(self, embeddings):
        all_embeddings = ["DE_TransE", "DE_DistMult", "DE_SimplE", "ATISE", "TERO", "TFLEX", "TimePlex"]
        return [e for e in all_embeddings if e in embeddings]
    
    def read_json(self, path):
        in_file = open(path, "r", encoding="utf8")
        dict = json.load(in_file)
        in_file.close()
        return dict
    
    def read_text(self, path):
        in_file = open(path, "r", encoding="utf8")
        text = in_file.read()
        in_file.close()
        return text

    def get_entity(self, measure):
        if "ENTITY" in measure.keys():
            return measure["ENTITY"]
        if "RELATION" in measure.keys():
            return measure["RELATION"]
        if "TIME" in measure.keys():
            return measure["TIME"]
    
    def round(self, val):
        return round(val, 1)
    
    def format_semester_9_hypothesis_1(self):
        embeddings = self.sort_methods(self.params.embeddings)
        metric = "MRR"

        prefix_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "semester_9_hypothesis_1_prefix.txt")
        suffix_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "semester_9_hypothesis_1_suffix.txt")
        shorthand_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "shorthand.json")

        prefix_text = self.read_text(prefix_path)
        suffix_text = self.read_text(suffix_path)
        shorthand = self.read_json(shorthand_path)

        for dataset in self.params.datasets:
            for split in self.params.splits:
                overall_scores_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "overall_scores.json")

                overall_scores = self.read_json(overall_scores_path)

                for normalized in ["", "_normalized"]:
                    if normalized == "_normalized":
                        continue

                    text = ""
                    highest_score = 0

                    for prediction_target in ["head", "relation", "tail", "time_from"]:
                        scores_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_9_hypothesis_1", prediction_target + normalized + ".json")

                        scores = self.read_json(scores_path)

                        text += r"\addplot coordinates { %" + prediction_target + "\n"
                        for i, embedding in enumerate(embeddings):
                            score = scores[embedding][metric] if embedding in scores.keys() else 0
                            text += f"({i}, {score}) %{embedding}" + "\n"
                            if score > highest_score:
                                highest_score = score
                        text += r"} ;" + "\n"
                    
                    for i, embedding in enumerate(embeddings):
                        overall_score = overall_scores[embedding][metric] if embedding in overall_scores.keys() else 0
                        text += r"\addplot[black,sharp plot,update limits=false,] coordinates { %" + embedding + "\n" + \
                        f"({float(i) - 0.5}, {overall_score})" + "\n" + \
                        f"({float(i) + 0.5}, {overall_score})" + "\n" + \
                        r"} ;" + "\n"

                    #max_y = min(highest_score*1.2, 1.0)
                    max_y = highest_score*1.2
                    
                    mod_prefix_text = prefix_text.replace(
                        "%1", f"""{",".join([shorthand[e] for e in embeddings])}""").replace(
                        "%2", str(max_y))
                    mod_suffix_text = suffix_text.replace(
                        "%1", f"{dataset}, split {split}").replace(
                        "%2", f"{dataset}_{split}")

                    output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "semester_9_hypothesis_1", dataset+"_"+split+normalized+".tex")
                    write(output_path, mod_prefix_text + text + mod_suffix_text)

    def format_hypothesis_2(self):
        for normalized in ["", "_normalized"]:
            for element_type in ["entity", "relation", "time"]:
                input_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_2", element_type +normalized + ".json")
                output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "hypothesis_2_"+element_type+normalized+".tex")
                
                input = self.read_json(input_path)

                min_val = 100.0
                max_val = -100.0

                result = \
                "\n" + \
                r"\begin{tabular}{r|RRRRRR}" + "\n" +\
                r"\multicolumn{1}{c|} {} &" + "\n" +\
                r"\multicolumn{1}{c} {DE-T} &" + "\n" +\
                r"\multicolumn{1}{c} {DE-D} &" + "\n" +\
                r"\multicolumn{1}{c} {DE-S} &" + "\n" +\
                r"\multicolumn{1}{c} {ATiSE} &" + "\n" +\
                r"\multicolumn{1}{c} {TeRo} &" + "\n" +\
                r"\multicolumn{1}{c} {TFLEX}\\ \hline" + "\n"
                    
                for i in range(0, 5):
                    result += self.get_entity(input[i]) +\
                    r" & " + str(self.round(input[i]["MEASURE"]["DE_TransE"]["MRR"])) +\
                    r" & " + str(self.round(input[i]["MEASURE"]["DE_DistMult"]["MRR"])) +\
                    r" & " + str(self.round(input[i]["MEASURE"]["DE_SimplE"]["MRR"]) ) +\
                    r" & " + str(self.round(input[i]["MEASURE"]["ATISE"]["MRR"])) +\
                    r" & " + str(self.round(input[i]["MEASURE"]["TERO"]["MRR"])) +\
                    r" & " + str(self.round(input[i]["MEASURE"]["TFLEX"]["MRR"])) + r"\\" + "\n"

                    for embedding in input[i]["MEASURE"].values():
                        val = self.round(embedding["MRR"])

                        if val < min_val:
                            min_val = val
                        if val > max_val:
                            max_val = val

                result += \
                r"\end{tabular}" + "\n"
                result = "\n" + r"\renewcommand{\MinNumber}{" + str(min_val) + r"}%" + "\n" +\
                r"\renewcommand{\MaxNumber}{" + str(max_val) + r"}%" + "\n" + result

                write(output_path, result)

    def format_embedding(self, embedding):
        if embedding == 'DE_TransE':
            return 'DE-T'
        if embedding == 'DE_DistMult':
            return 'DE-D'
        if embedding == 'DE_SimplE':
            return 'DE-S'
        if embedding == 'TERO':
            return 'TeRo'
        if embedding == 'ATISE':
            return 'ATiSE'
        if embedding == 'TFLEX':
            return 'TFLEX'

    def get_overlap(self, overlaps, emb_n, emb_m):
        for o in overlaps:
            if o["EMBEDDING_N"] == emb_n and o["EMBEDDING_M"] == emb_m:
                return o["OVERLAP_TOP"]

    def to_str(self, value):
        return "{:.1f}".format(value, 1)

    def format_hypothesis_2_overlap(self):
        for element_type in ["entity", "relation", "time"]:
            input_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_2", "top_x_overlap", element_type + "_top_20_overlap.json")
            output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "hypothesis_2_" + element_type + "_top_20_overlap.tex")
            
            overlaps = self.read_json(input_path)

            min_val = 100.0
            max_val = -100.0

            result = \
            "\n" + \
            r"\begin{tabular}{r|RRRRRR}" + "\n" +\
            r"\multicolumn{1}{c|} {} &" + "\n" +\
            r"\multicolumn{1}{c} {DE-T} &" + "\n" +\
            r"\multicolumn{1}{c} {DE-D} &" + "\n" +\
            r"\multicolumn{1}{c} {DE-S} &" + "\n" +\
            r"\multicolumn{1}{c} {ATiSE} &" + "\n" +\
            r"\multicolumn{1}{c} {TeRo} &" + "\n" +\
            r"\multicolumn{1}{c} {TFLEX}\\ \hline" + "\n"
                
            for embedding_n in ['DE_TransE', 'DE_DistMult', 'DE_SimplE', 'ATISE', 'TERO', 'TFLEX']:
                result += self.format_embedding(embedding_n)
                for embedding_m in ['DE_TransE', 'DE_DistMult', 'DE_SimplE', 'ATISE', 'TERO', 'TFLEX']:
                    if embedding_n == embedding_m:
                        result += r" & \multicolumn{1}{c}{100.0}"
                    else:
                        result += r" & " + self.to_str(self.round(self.get_overlap(overlaps, embedding_n, embedding_m)*100.0))
                result += r"\\" + "\n"

            for overlap in overlaps:
                if overlap["EMBEDDING_N"] == overlap["EMBEDDING_M"]:
                    continue

                val = self.round(overlap["OVERLAP_TOP"]*100.0)

                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val

            result += \
            r"\end{tabular}" + "\n"
            result = "\n" + r"\renewcommand{\MinNumber}{" + str(min_val) + r"}%" + "\n" +\
            r"\renewcommand{\MaxNumber}{" + str(max_val) + r"}%" + "\n" + result

            write(output_path, result)    

    def format_hypothesis_3(self):
        for normalized in ["", "_normalized"]:
            input_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_3", "hypothesis_3" + normalized + ".json")
            output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "hypothesis_3" + normalized + ".tex")
            
            input = self.read_json(input_path)

            min_val = 100.0
            max_val = 0.0

            result = \
            "\n" + \
            r"\begin{tabular}{r|l|SSSSSS}" + "\n" +\
            r"$e_n$ &" + "\n" +\
            r"$e_m$ &" + "\n" +\
            r"\multicolumn{1}{c} {DE-T} &" + "\n" +\
            r"\multicolumn{1}{c} {DE-D} &" + "\n" +\
            r"\multicolumn{1}{c} {DE-S} &" + "\n" +\
            r"\multicolumn{1}{c} {ATiSE} &" + "\n" +\
            r"\multicolumn{1}{c} {TeRo} &" + "\n" +\
            r"\multicolumn{1}{c} {TFLEX}\\ \hline" + "\n"
            i = 0
            num_of_rows = 0
            while num_of_rows < 5:
                if "DIFFERENCE" in input[i].keys():
                    result += input[i]["ENTITY_N"] + r" & " + input[i]["ENTITY_M"] +\
                    r" & " + str(self.round(input[i]["DIFFERENCE"]["DE_TransE"])) +\
                    r" & " + str(self.round(input[i]["DIFFERENCE"]["DE_DistMult"])) +\
                    r" & " + str(self.round(input[i]["DIFFERENCE"]["DE_SimplE"]) ) +\
                    r" & " + str(self.round(input[i]["DIFFERENCE"]["ATISE"])) +\
                    r" & " + str(self.round(input[i]["DIFFERENCE"]["TERO"])) +\
                    r" & " + str(self.round(input[i]["DIFFERENCE"]["TFLEX"])) + r"\\" + "\n"

                    for embedding in input[i]["DIFFERENCE"].keys():
                        val = self.round(input[i]["DIFFERENCE"][embedding])

                        if val < min_val:
                            min_val = val
                        if val > max_val:
                            max_val = val
                    
                    num_of_rows += 1
                i += 1
            
            result += \
            r"\end{tabular}" + "\n"
            result = "\n" + r"\renewcommand{\MinNumber}{" + str(min_val) + r"}%" + "\n" +\
            r"\renewcommand{\MaxNumber}{" + str(max_val) + r"}%" + "\n" + result

            write(output_path, result)

    def format_no_of_entities(self):
        for element in ["entities", "relations", "timestamps"]:
            input_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "no_of_elements", "train_" + element + ".json")
            output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "no_of_elements_train_" + element + ".tex")
            
            input = self.read_json(input_path)

            min_val = 999
            max_val = 0

            result = r"\addplot+ coordinates {" + "\n"
            for i, e in enumerate(input):
                val = e["COUNT"]
                result += r"   (" + str(i) + r", " + str(val) + r")" + "\n"
                
                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val
            
            result += r"} ;"
            result = r"% MIN VAL: " + str(min_val) + "\n" + r"% MAX VAL: " + str(max_val) + "\n\n" + result

            write(output_path, result)

    def format(self):
        #self.format_hypothesis_2()
        #self.format_hypothesis_3()
        #self.format_no_of_entities()
        # self.format_hypothesis_2_overlap()
        # self.format_semester_9_hypothesis_1()
        # format_relation_property = FormatRelationPropertyHypothesis(self.params)
        # format_relation_property.format()
        # format_voting_hypothesis = FormatVotingHypothesis(self.params)
        # format_voting_hypothesis.format()

        for t in self.task:
            tex = texobject(self.params, t)
            match(t):
                case "time_prediction_mae":
                    tex.caption = "MAE of model prediction. "\
                                "Values are given in days for ICEWS14 and years for WikiData12k and YAGO11k. "\
                                "Where the prediction is a timespan the average is given as '\\textsc{BEST}\u2013\\textsc{WORST}'"
                    tex.format()

                case "time_prediction_distribution":
                    for method in self.params.embeddings:
                        for dataset in self.params.datasets:
                            tex.caption = f"Distribution of predictions on timestamps for {method} on {dataset}"
                            tex.type = "fig"
                            tex.embeddings = method
                            tex.datasets = dataset
                            tex.format()
                
                case "time_error_distibution":
                    format_error_distribution = FormatErrorDistribution(self.params)
                    format_error_distribution.format()
