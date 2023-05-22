import os
import builtins
import re
import glob
from scripts import read_json, read_text, write


class texobject():    
    def __init__(self, params, task):
        self.params = params
        self.datasets = self.params.datasets
        self.embeddings = self.params.embeddings
        self.splits = self.params.splits
        self.task = task
        self.type = "tab"
        self.width = "column"
        self.caption = self.xlabel = self.ylabel = "\\missing"
        self.foreach = False

    def format(self):
        # paths
        template_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "tex_template.tex")
        shorthand_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "method_shorthand.json")
        output_path = os.path.join(self.params.base_directory, "formatlatex", "result")
        if self.foreach: output_path = os.path.join(output_path, self.task)

        # vars
        template = self.read_template(template_path)
        self.shorthand = read_json(shorthand_path)
        self.output_filename = self.task + ".tex" # can be edited in construct_rows
        self.label = self.type + ":" + self.task # read label or default to type:task, append to in construct_rows
        self.rows = getattr(self, f'construct_rows_{self.task}')() # func depends on task

        # format content
        content = getattr(self, f'format_content_{self.type}')(self.rows)

        # replace text in template
        for pair in (
            ("_caption", self.caption),
            ("_label", self.label),
            ("_content", content),
            ("_method", self.mkstr(self.embeddings)),
            ("_dataset", self.mkstr(self.datasets)),
            ("_xlabel", self.xlabel),
            ("_ylabel",  self.ylabel),
        ):
            template = re.sub(pair[0], repr(pair[1])[1:-1], template)
        output = template

        
        # write to file
        write(os.path.join(output_path, self.output_filename), output)

###############################################    FORMAT TEXT FUNCTIONS     ################################################

    def format_content_tab(self, rows):
        content = ""
        columns = "l" + (len(rows[0][0])-1)*"c" # set latex columns
        highlight_results = self.find_best_results(rows, min)

        content += f"\\begin{{tabular}}{{{columns}}}\n"

        for sec in rows:
            content += "\\hline\n" #each section separated by hlines
            for row in sec:
                row = self.format_values(row, self.shorthand) # decimal precision and shorthands
                row = self.highlight_values(row, self.format_values(highlight_results, self.shorthand)) # highlight best results
                row = self.flatten_row(row) # flatten rows with sublists, join values by en-dash
                content += self.tex_row(row) # format row as row in tex table
        content += "\\hline\n"
        content += "\end{tabular}\n"

        return content
    
    def format_content_fig(self, rows):
        content = ""

        properties = {
            "COLOR": "tikzdarkblue",
            "PATTERN": "north east lines",
            "PATTERN COLOR": "tikzlightblue",
            "OPACITY": "0.8"
        }

        for row in rows:
            properties.update(row[1])
            content += f"\\addplot["
            for key in properties.keys():
                content+=f"\n{key}={properties[key]},"
            content += f"]\ntable {{{row[0]}}};\n"

        return content

    # format list as row in tex table
    def tex_row(self, row):
        return str(" & ".join(row) + "\\\\\n")
    
    # format text as bold for tex
    def tex_bold(self, x):
        return f"\\textbf{{{x}}}"
    
    # format all items in list of lists
    def format_values(self, input, shorthand):
        if type(input) == list: #make this pretty someday
            result = input.copy()
        else:
            result = [input]

        for i, item in enumerate(result):
            match type(item):
                case builtins.list:
                    result[i] = self.format_values(item, shorthand) # recurse sublists
                case builtins.int | builtins.float | builtins.complex:
                    result[i] = f"{item:.2f}" # .2 decimal precision
                case builtins.str:
                    if item in shorthand.keys():
                        result[i] = str(shorthand[item]) # format as shorthand only if string in shorthand list
        return result

    # flatten single row, join values of sublists by en-dash
    def flatten_row(self, row):
        flat_row = []

        for item in row:
            if type(item) == list:
                item = str("\u2013".join(item))
            flat_row.append(item)

        return flat_row
    
    def mkstr(self, input):
        return str(", ".join(self.format_values(input, self.shorthand)))

    # match row against test, if values and positions match transform row value by highlight, return row
    def highlight_values(self, row, test, highlight = "tex_bold"):
        result = []

        for x, y in enumerate(row):
            if type(y) == list:
                y = self.highlight_values(y, [test[x]]*len(y))
            elif y == test[x]:
                y = getattr(self, highlight)(y)
            result.append(y)

        return result

########################################################     ROW OPERATIONS     ##########################################################

    # read template, adjust if tablewidth is "text"
    def read_template(self, path):
        template = read_text(path)

        if self.width == "text":
            template = template.replace(
                "textype", "textype*").replace(
                "columnwidth", "textwidth"
                )

        match(self.type):
            case "fig":
                template = template.replace("_textype", "figure")
                template = re.sub(r'(\\caption.*)\n(\\vspace.*)\n*(_content)\n*',
                              '\n\g<3>\n\n\g<2>\n\g<1>\n', 
                              template)
                template = template.replace("centering\n\n", "centering\n\\begin{tikzpicture}\n\\begin{axis}[\nno markers,\nxlabel={_xlabel},\nylabel={_ylabel}]\n\n")
                template = template.replace("\n\\vspace", "\n\\end{axis}\n\\end{tikzpicture}\n\\vspace")
            case "tab":
                template = template.replace("_textype", "table")

        return template


    # get columns of list of rows
    def get_columns(self, input):
        rows = [row for sec in input for row in sec] # flatten list for sections
        columns = [[row[i] for row in rows] for i in range(len(rows[0]))] # get columns
        return columns

    # find best result for each column, append to list, return list
    def find_best_results(self, input, minmax = max):
        columns = self.get_columns(input)
        best_results = []

        for column in columns:
            best_results.append(self.find_best_result_of_column(column, minmax))

        return best_results

    # finds best result per column
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
            
####################################################     CONSTRUCT ROWS FUNCTIONS    ################################################
    # give error if rows are not same length
    def row_length_error(self, rows):
        if (len(set([len(row) for sec in rows for row in sec]))) != 1:
            print("ERROR: Length of rows differs")

    # construct rows for average difference between best and correct prediction
    def construct_rows_time_prediction_mae(self):
        # read input
        input = {}
        for dataset in self.datasets:
            input_path = os.path.join(self.params.base_directory, "result", dataset, "split_original", "timestamp_prediction_avg.json")
            input[dataset] = read_json(input_path)

        # construct rows (rows have sections, hline between each section)
        rows = []

        firstrow = [["Methods"] + self.datasets]
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

        self.row_length_error(rows)

        return(rows)
    
    # construct rows for distribution of time prediction errors
    def construct_rows_time_prediction_distribution(self):
        rows = []
        id = f"{self.datasets}_{self.embeddings}"
        self.output_filename = f"{id}.tex"
        self.label += f"_{id}"

        # edit file
        filename = f"{self.embeddings}*"
        input_path = os.path.join(self.params.base_directory, "result", self.datasets, "split_" + self.splits, self.task, filename)
        
        for path in glob.glob(input_path):
            id = f"{self.datasets}_{self.embeddings}"
            properties = {}
            diff = re.split(rf'{filename[:-1]}_?(.*)\.dat', path)[1] # extract best/wors part of filename
            match(diff):
                case "best":
                    properties.update({"COLOR": "green"})
                    id += f"_{diff}"
                case "wors":
                    properties.update({"COLOR": "red"})
                    properties.update({"PATTERN": "north west lines"})
                    id += f"_{diff}"
                    diff += "t"

            rows.append([f"content/appendix/figures/{self.task}/{id}.dat", properties])

        return rows