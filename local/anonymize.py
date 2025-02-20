# Copyright 2020 Unibg Seclab (https://seclab.unibg.it)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json
import time

import pandas as pd

from mondrian import generalization as gnrlz
from mondrian.anonymization import anonymize
from mondrian.evaluation import discernability_penalty
from mondrian.evaluation import global_certainty_penalty
from mondrian.evaluation import normalized_certainty_penalty
from mondrian.score import entropy, neg_entropy, span
from mondrian.visualization import visualizer
from mondrian.test import result_handler


def __generalization_preproc(job, df):
    """Anonymization preprocessing to arrange generalizations.

    :job: Dictionary job, contains information about generalization methods
    :df: Dataframe to be anonymized
    :returns: Dictionary of taxonomies required to perform generalizations
    """
    if 'quasiid_generalizations' not in job:
        return None

    quasiid_gnrlz = dict()

    for gen_item in job['quasiid_generalizations']:
        g_dict = dict()
        g_dict['qi_name'] = gen_item['qi_name']
        g_dict['generalization_type'] = gen_item['generalization_type']
        g_dict['params'] = gen_item['params']

        if g_dict['generalization_type'] == 'categorical':
            # read taxonomy from file
            t_db = g_dict['params']['taxonomy_tree']
            if t_db is None:
                raise gnrlz.IncompleteGeneralizationInfo()
            taxonomy = gnrlz._read_categorical_taxonomy(t_db)
            # taxonomy.show()
            g_dict['taxonomy_tree'] = taxonomy
        elif g_dict['generalization_type'] == 'numerical':
            try:
                fanout = g_dict['params']['fanout']
                accuracy = g_dict['params']['accuracy']
                digits = g_dict['params']['digits']
            except KeyError:
                raise gnrlz.IncompleteGeneralizationInfo()
            if fanout is None or accuracy is None or digits is None:
                raise gnrlz.IncompleteGeneralizationInfo()
            taxonomy, minv = gnrlz.__taxonomize_numeric(
                df=df,
                col_label=g_dict['qi_name'],
                fanout=int(fanout),
                accuracy=float(accuracy),
                digits=int(digits))
            g_dict['taxonomy_tree'] = taxonomy
            g_dict['min'] = minv
            # taxonomy.show()
            # print("Minv: {}".format(minv))
        # elif g_dict['generalization_type'] == 'common_prefix':
        # common_prefix generalization doesn't require taxonomy tree

        quasiid_gnrlz[gen_item['qi_name']] = g_dict

    # return the generalization dictionary
    return quasiid_gnrlz


def main():
    parser = argparse.ArgumentParser(
        description='Anonymize a dataset using Mondrian.')
    parser.add_argument('METADATA', help='json file that describes the job.')
    parser.add_argument('DEMO',
                        default=0,
                        type=int,
                        help='Launch in demo mode.')

    args = parser.parse_args()
    demo = args.DEMO

    if demo == 1:
        print("\n[*] Read configuration file")
        input("\t Press any key to continue...")

    with open(args.METADATA) as fp:
        job = json.load(fp)

    start_time = time.time()
    # Measures
    test_measures = {}

    # Parameters
    input = job['input']
    output = job['output']
    id_columns = job.get('id_columns', [])
    redact = job.get('redact', False)
    quasiid_columns = job['quasiid_columns']
    sensitive_columns = job.get('sensitive_columns', [])
    # when column score is not given it defaults to span
    score_functions = {'span': span,
                       'entropy': entropy,
                       'neg_entropy': neg_entropy}
    if 'column_score' in job and job['column_score'] in score_functions:
        column_score = score_functions[job['column_score']]
    else:
        column_score = span
    K = job.get('K')
    L = job.get('L')
    measures = job.get('measures', [])

    if K:
        test_measures["K"] = K

    if not K and not L:
        raise Exception("Both K and L parameters not given or equal to zero.")
    if L:
        test_measures["L"] = L
        if not sensitive_columns:
            raise Exception(
                "l-diversity needs to know which columns are sensitive."
            )

    if demo == 1:
        print("\n[*] Job info configured")
        input("\t Press any key to continue...")

    if demo == 1:
        print("\n[*] Reading the dataset")
    df = pd.read_csv(input)
    print(df.head)

    quasiid_gnrlz = __generalization_preproc(job, df)

    if demo == 1:
        print("\n[*] Taxonomies info read")
        input("\t Press any key to continue...\n")

    qi_range = [-1] * len(quasiid_columns)
    for i, column in enumerate(quasiid_columns):
        qi_range[i] = span(df[column])

    adf = anonymize(
        df=df,
        id_columns=id_columns,
        redact=redact,
        quasiid_columns=quasiid_columns,
        sensitive_columns=sensitive_columns,
        column_score=column_score,
        K=K,
        L=L,
        quasiid_gnrlz=quasiid_gnrlz)

    if demo == 1:
        print("\n[*] Dataset anonymized")
        input("\t Press any key to continue...")

    print('\n[*] Anonymized dataframe:\n')

    if adf.size < 50:
        print(adf)
        visualizer(adf, quasiid_columns)
    else:
        print(adf.head)

    if demo == 1 and measures:
        print("\n[*] Starting evaluate information loss")
        input("\t Press any key to continue...")

    if measures:
        print('\n[*] Information loss evaluation\n')
    for measure in measures:
        if measure == 'discernability_penalty':
            dp = discernability_penalty(adf, quasiid_columns)
            print(f"Discernability Penalty = {dp:.2E}")
            test_measures["DP"] = dp
        elif measure == 'normalized_certainty_penalty':
            ncp = normalized_certainty_penalty(df, adf, quasiid_columns,
                                               qi_range, quasiid_gnrlz)
            print(f"Normalized Certainty Penalty = {ncp:.2E}")
            test_measures["NCP"] = ncp
        elif measure == 'global_certainty_penalty':
            gcp = global_certainty_penalty(df, adf, quasiid_columns,
                                           qi_range, quasiid_gnrlz)
            print(f"Global Certainty Penalty = {gcp:.4f}")
            test_measures["GCP"] = gcp
    # Write file according to extension
    print(f"\n[*] Writing to {output}")
    adf.to_csv(output, index=False)

    print('\n[*] Done\n')
    end_time = time.time()
    if demo == 0:
        print("--- %s seconds ---" % (end_time - start_time))
    test_measures["time"] = (end_time - start_time)
    test_measures["timestamp"] = end_time
    result_handler(test_measures)

if __name__ == "__main__":
    main()
