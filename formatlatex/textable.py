
import os
import builtins
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
        shorthand_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "method_shorthand.json")
        output_path = os.path.join(self.params.base_directory, "formatlatex", "result", self.task + ".tex")

        # table vars
        template = read_text(template_path)
        shorthand = read_json(shorthand_path)
        rows = getattr(self, f'construct_rows_{self.task}')()
        columns = "l" + (len(rows[0][0])-1)*"c"
        label = "tab:" + self.task

        col_best_results = self.find_best_results(rows, min)
        rows = self.format_values(rows, shorthand)

        # TO DO LIST
        # highlight best result
        # flatten sublist items f"{item['BEST']:.2f}\u2013{item['WORST']:.2f}"

#        # format content
#        content = ""
#        for sec in rows:
#            content += "\\hline\n"
#            for row in sec:
#                content += self.tex_row(row)
#        content += "\\hline\n"
#
#        # replace text in template
#        output = template.replace(
#            "_caption", self.caption).replace(
#            "_label", label).replace(
#            "_columns", columns).replace(
#            "_content", content)
        
        # write to file
        #print(output)
        #write(output_path, output)

    def tex_row(self, row):
        return str(" & ".join(row) + "\\\\\n")

    # find best result for each column, append to list, return list
    def find_best_results(self, input, minmax = max):
        rows = [row for sec in input for row in sec] # flatten list for sections
        columns = [[row[i] for row in rows] for i in range(len(rows[0]))] # get columns
        best_results = []

        for column in columns:
            best_results.append(self.find_best_result_of_column(column, minmax))

        return best_results

    def find_best_result_of_column(self, input, minmax = max):
        # return value, stays unchanged if input does not contain numbers
        best_result = False

        # check if items in lists are lists themselves, recurses if yes
        for i, item in enumerate(input):
            if type(item) == list:
                input[i] = self.find_best_result_of_column(item, minmax)

        # filters non-number values (e.g. header of column)
        filtered_column = list(filter( lambda x: type(x) in {int, float, complex}, input))

        # finds best of numbers in column, best defined by minmax parameter
        if bool(filtered_column) == True:
            best_result = (minmax(filtered_column))

        return best_result

    # format all items in input
    def format_values(self, input, shorthand):
        for i, item in enumerate(input):
            match type(item):
                case builtins.list:
                    input[i] = self.format_values(item, shorthand)
                case builtins.int | builtins.float | builtins.complex:
                    input[i] = f"{item:.2f}"
                case builtins.str:
                    if item in shorthand.keys():
                        input[i] = str(shorthand[item])
        return input
            

    def construct_rows_temporal_precision_avg_diff(self):
        # read input
        input = {}
        for dataset in self.datasets:
            input_path = os.path.join(self.params.base_directory, "result", dataset, "split_original", "timestamp_prediction_avg.json")
            input[dataset] = read_json(input_path)

        # construct rows (rows have sections, hline between each section)
        rows = []

        firstrow = [["MET"] + self.datasets]
        rows.append(firstrow)

        avgs = []
        for embedding in self.embeddings:
            tmp_row = []

            # embedding name
            tmp_row += [embedding]

            # load averages
            for item in [input[d][embedding] for d in self.datasets]:
                if type(item) == dict:
                    item = list(item.values())
                tmp_row += [item]

            # append line
            avgs.append(tmp_row)
        rows.append(avgs)

        # give error if rows are not same length
        if (len(set([len(row) for sec in rows for row in sec]))) != 1:
            print("ERROR: Length of rows differs")

        return(rows)
