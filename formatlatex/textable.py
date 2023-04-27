
import os
import json
from scripts import read_json, read_text, write


class TEXTable():    
    def __init__(self, params, task, tablewidth = "column", caption = "\\missing"):
        self.params = params
        self.datasets = self.params.datasets
        self.embeddings = self.params.embeddings
        self.task = task
        self.tablewidth = tablewidth
        self.caption = caption


    def format(self):
        # paths
        template_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "table_template_" + self.tablewidth + ".tex")
        output_path = os.path.join(self.params.base_directory, "formatlatex", "result", self.task + ".tex")

        # table vars
        template = read_text(template_path)
        rows = getattr(self, f'construct_rows_{self.task}')()
        columns = "l" + (len(rows[0][0])-1)*"c"
        label = "tab:" + self.task

        # format content
        content = ""
        for sec in rows:
            content += "\\hline\n"
            for row in sec:
                content += self.tex_row(row)  
        content += "\\hline\n"

        # replace text in template
        output = template.replace(
            "_caption", self.caption).replace(
            "_label", label).replace(
            "_columns", columns).replace(
            "_content", content)
        
        # write to file
        print(output)
        #write(output_path, output)

    def tex_row(self, list):
        return str(" & ".join(list) + "\\\\\n")

    def position_of_best_result(self, list):
        position = ''

        print(position)


    def construct_rows_temporal_precision_avg_diff(self):
        # load correctly formatted names of embeddings and datasets
        shorthand_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "method_shorthand.json")
        shorthand = read_json(shorthand_path)

        # read input
        input = {}
        for dataset in self.datasets:
            input_path = os.path.join(self.params.base_directory, "result", dataset, "split_original", "timestamp_prediction_avg.json")
            input[dataset] = read_json(input_path)

        # construct rows (rows have sections, hline between each section)
        rows = []

        rows.append([["Methods"] + [shorthand[d] for d in self.datasets]])

        avgs = []
        for embedding in self.embeddings:
            tmp_row = []
            emb_results = [input[d][embedding] for d in self.datasets]

            # embedding name
            tmp_row += [shorthand[embedding]]

            # load averages
            for item in emb_results:
                if type(item) == float:
                    item = f"{item:.2f}"
                if type(item) == dict:
                    item = f"{item['BEST']:.2f}\u2013{item['WORST']:.2f}"
                tmp_row += [str(item)]

            # append line
            avgs.append(tmp_row)
        rows.append(avgs)

        # give error if rows are not same length
        if (len(set([len(row) for sec in rows for row in sec]))) != 1:
            print("ERROR: Length of rows differs")

        return(rows)
