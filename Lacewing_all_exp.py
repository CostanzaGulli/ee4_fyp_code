import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
import seaborn as sns
sns.set_theme(style='whitegrid')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score

import Lacewing_single_exp_algo as la

if __name__ == "__main__":
    current_path = Path('..', 'data_files')
    experiments = [x for x in current_path.iterdir() if x.is_dir()]
    excel = True
    if excel:
        # load spreadsheet with actual results
        excel_df = pd.read_excel(r'../data_files/20200511_Notebook_Experiments_Samples.xls', sheet_name='Sheet1')
        excel_df['ID'] = excel_df['ID'].astype(str)
        excel_df.set_index('ID', inplace=True)

        all_exp_summary = pd.DataFrame()
        positive_true = pd.DataFrame()
        positive_pred = pd.DataFrame()
        all_qubit = pd.DataFrame()
        all_count = pd.DataFrame()
        all_perc = pd.DataFrame()
        all_fiterror = pd.DataFrame()

        for exp_path in tqdm(experiments):

            # Get string of experiment ID
            exp_path_str = str(exp_path)
            experiment_id = exp_path_str[exp_path_str.rfind('_') + 1:]

            if experiment_id not in excel_df.index:
                print(f'Could not find experiment {experiment_id} in the spreadsheet')
                continue

            # Get data for this experiment from spreadsheet
            qubit_control = excel_df.loc[experiment_id, 'Qubit Control']
            qubit_sample = excel_df.loc[experiment_id, 'Qubit Sample']

            assert not isinstance(qubit_control, pd.Series), f'Multiple entries found in {experiment_id}'

            QUBIT_THRESH = 200
            if qubit_control >= QUBIT_THRESH:
                positive_true = positive_true.append(['positive'], ignore_index=True)
            else:
                positive_true = positive_true.append(['negative'], ignore_index=True)

            if qubit_sample >= QUBIT_THRESH:
                positive_true = positive_true.append(['positive'], ignore_index=True)
            else:
                positive_true = positive_true.append(['negative'], ignore_index=True)

            out = la.algo(exp_path, visualise_preprocessing=False, visualise_processing=False)
            # all_exp_summary.extend(out)
            wells_summary = all_exp_summary.append(out['well data'], ignore_index=True)
            positive_pred = positive_pred.append([out['well data']['Top Well']['result']], ignore_index=True)
            positive_pred = positive_pred.append([out['well data']['Bot Well']['result']], ignore_index=True)

            all_qubit = all_qubit.append([qubit_control], ignore_index=True)
            all_qubit = all_qubit.append([qubit_sample], ignore_index=True)

            all_count = all_count.append([out['well data']['Top Well']['pos count']], ignore_index=True)
            all_count = all_count.append([out['well data']['Bot Well']['pos count']], ignore_index=True)
            all_perc = all_perc.append([out['well data']['Top Well']['pos percentage']], ignore_index=True)
            all_perc = all_perc.append([out['well data']['Bot Well']['pos percentage']], ignore_index=True)
            all_fiterror = all_fiterror.append([out['well data']['Top Well']['max fit error at infl']], ignore_index=True)
            all_fiterror = all_fiterror.append([out['well data']['Bot Well']['max fit error at infl']], ignore_index=True)

            # for item in out:
                # print(item['table df'], end='\n')

        # fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
        # ax[0].scatter(all_qubit, all_count)
        # ax[0].set(title='Number of pixels with fitting error above threshold\nFitting until 1 9min',
        #           xlabel='Qubit value', ylabel='Number of pixels above threshold')
        # ax[1].scatter(all_qubit, all_perc)
        # ax[1].set(title='Percentage of active pixels with fitting error above threshold\nFitting until 19min',
        #           xlabel='Qubit value', ylabel='Percentage of active pixels above threshold')
        # plt.savefig('plot_withqubit19.eps')
        # plt.show()

        print(f'SIZES {all_qubit.shape, all_fiterror.shape}')
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=100)
        ax.scatter(all_qubit, all_fiterror)
        ax.set(title='Max error at inflection point',
                  xlabel='Qubit value', ylabel='Max fit error')
        plt.show()

        cm = confusion_matrix(positive_true, positive_pred, labels=['positive', 'negative', 'inconclusive'], normalize=None)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Positive', 'Negative', 'Inconclusive'])
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
        disp.plot(ax=ax, cmap='cividis')
        ax.grid(False)
        # plt.savefig('conf_matrix.eps')
        plt.show()

        accuracy = accuracy_score(positive_true, positive_pred)
        print(f'The accuracy is {accuracy}')

        f1score = f1_score(positive_true, positive_pred, average="micro", pos_label='positive')
        print(f'The f1-score is {f1score}')
    else:
        for exp_path in tqdm(experiments):
            out = la.algo(exp_path, visualise_preprocessing=True, visualise_processing=False)