from experiment import experiment_template
import csv


class Blank(experiment_template.Experiment):

    def run(self):
        self.info("Hello world, I just wanna see everything's working fine.")
        x = ['John Smith', 'Accounting', 'November']
        y = ['Erica Meyers', 'IT', 'March']
        self.write_summary_header(["name", "job", "month"])
        self.write_summary_content(x)
        self.write_summary_content(y)
