from recbole.quick_start import run_recbole



parameter_dict = {
   'neg_sampling': None,
}
config_file_list = ['sas_cts_2.yaml']
run_recbole(model='SASCTS', dataset='ml-1m', config_file_list=config_file_list, config_dict=parameter_dict)
