import os
import argparse

import matplotlib
import matplotlib.pyplot as plt

import plotting as plotting

def master_plots(df, header, ymin, ymax, savepath, experiment_name,
                 pptx_path=None,
                 show=False, verbose=False):
    
    fig, ax, removed = plotting.create_boxplot(df, header, ymin, ymax,
                                               experiment_name)
    # create folder specific for other stuff
    extra_savepath = f"{savepath}extra_information/"
    os.makedirs(extra_savepath, exist_ok=True)
    #save removed data
    removed.to_csv(f"{extra_savepath}{header}_removed.csv", index=True)
    if verbose:
        print(f"{len(removed)} removed data points for {header}")
    # plot
    fig.savefig(f"{savepath}{header}.png")
    if show: 
        plt.show()
        plt.close()

    # we also add to a powerpoint 
    if pptx_path:
        plotting.add_to_powerpoint(fig, f"{header} Analysis", pptx_path, 
                                   experiment_name, date)

    matplotlib.pyplot.close()

def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--date", help="", type=str, required=True)
    parser.add_argument("--specific_savepath", help="In case you need to save elsewhere than the default location", 
                        type=str, required=False)
    parser.add_argument("--ordering", help="", type=str, required=False,
                        default="None")
    parser.add_argument("--sep_category", help="", type=str,required=False,
                        default="$")
    parser.add_argument("--sep_count", help="", type=str,required=False,
                        default="%")
    parser.add_argument("--summary_only", action="store_true",
                    help="Only output the summary",
                    default=False)
    parser.add_argument("--root", help="", type=str, required=False,
                        default=f"/Users/michaelmoret/Library/CloudStorage/GoogleDrive-michael@externa.bio/.shortcut-targets-by-id/1BdUNsBjDh5Gee_76jCiKB1C_CwG0ercP")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    date = args.date

    # tmp constant 
    order = None
    dev = False
    y_none = True

    for MODE in ["triple", "single"]:
        print(f"Processing {MODE} data")
        pathfile = f"Pulling data/{date}/{MODE}/"
        root = f"{args.root}/{pathfile}"

        # if root does not exist, skip script
        if not os.path.exists(root):
            print(f"Path {root} does not exist")
            continue

        # excel master file to log the summary
        # shared with oddity
        if not args.specific_savepath:
            path_excel_master = f"{args.root}/Pulling data/1_experiment_summary_with_time.xlsx"
            path_pptx_master = f"{args.root}/Pulling data/1_break_stress.pptx"
        elif args.specific_savepath:
            path_excel_master = f"{args.specific_savepath}temp_hair_summary.xlsx"
            path_pptx_master = f"{args.specific_savepath}temp_break_stress.pptx"

        all_dfs = []
        all_fns = []
        all_exp_names = []

        # get the data and apply the cleaning
        for file in os.listdir(root):
            if file.endswith(".txt"):
                print("\n*****************************")
                print("file: ", file)
                
                # clean the df
                if MODE == "triple":
                    df = plotting.get_df_from_file(root + file,
                                                   skip=2)
                    df = plotting.clean_triple(df, max_100=True)
                elif MODE == "single":
                    df = plotting.get_df_from_file(root + file,
                                                   skip=10)
                    df = plotting.clean_single(df, max_100=True)
                else:
                    raise ValueError(f"Not a {MODE} experiment")

                splitted_name = file.split(args.sep_category)
                # get the treatment time
                treatment_time = splitted_name[1]
                df["Treatment time"] = treatment_time
                # get the hair name and add it in the df
                hair_name = splitted_name[2]
                df["Hair"] = hair_name
                # get the control condition
                control = ""
                for name in splitted_name:
                    if "control" in name.lower() or "ctrl" in name.lower():
                        if "aa" in name.lower():
                            control = "Amonium acetate"
                        elif "phos" in name.lower():
                            control = "Phosphate"
                df["Treatment"] = control

                # get the experiments; i.e. not the date
                # not the single or triple etc
                splitted_name = splitted_name[3:-1]
                print("splitted_name: ", splitted_name)
                # add the experiment name in the dataframe
                all_names = []
                for entries in splitted_name:
                    
                    times_name = entries.split(args.sep_count)
                    _times = int(times_name[0])
                    _name = times_name[1]
                    all_names += [_name] * _times
                print(f"len df: {len(df)}, len names: {len(all_names)}")

                if order is not None:
                    # check if we have the order
                    # get unique names
                    _unique_names = list(set(all_names))
                    # make sure all names match with order
                    for name in _unique_names:
                        if name not in order:
                            print("")
                            print(f"{name} not in order")
                            print("Unique names: ", _unique_names)
                            print("Order: ", order)                

                df["Name"] = all_names
                # remove nan
                df = df.dropna()  
                # add a column with the date
                df["Date"] = date
                # reorder the dataframes if necessary
                if order is not None:
                    print("reordering")
                    df = plotting.reorder_by_names(df, order)
                
                all_dfs.append(df)
                all_fns.append(file.replace(".txt", ""))
                all_exp_names.append(f"{date} | {hair_name} | {treatment_time}")

        # plotting triple sepcific plots
        if MODE == "triple":
            print("\n\nUpdating the master excel file")
            print("Master file name: ", path_excel_master)
            try:
                for df, name in zip(all_dfs, all_fns):
                    plotting.write_summary_stats(df, name, master_file=path_excel_master)
            except Exception as e:
                print("Filename problem: ", file)
                print(f"Error updating master file: {e}")
            
            if not args.summary_only:
                print("\nplotting triple data")
                for df, name, experiment_name in zip(all_dfs, all_fns, all_exp_names):
                    print("\n************************************")
                    print(f"{name}\n")
                    print(f"df length: {len(df)}")
                    try:
                        if args.specific_savepath is None:
                            savepath = f"{root}{name}/"
                        else:
                            savepath = args.specific_savepath
                        if dev:
                            savepath = f"{root}dev/"
                        os.makedirs(savepath, exist_ok=True)
                        
                        try:
                            header = 'MEAN DIAMETER'
                            if y_none:
                                ymin = None
                                ymax = None
                            else:
                                ymin = 20
                                ymax = 120
                            master_plots(df, header, ymin, ymax, savepath, experiment_name)
                        except Exception as e:
                            print(f"ERROR with {header}")
                            print("Error:", e)

                        try:
                            header = 'BREAK STRESS'
                            if y_none:
                                ymin = None
                                ymax = None
                            else:
                                ymin = 120
                                ymax = 280
                            master_plots(df, header, ymin, ymax, savepath, experiment_name, 
                                         path_pptx_master)
                        except Exception as e:
                            print(f"ERROR with {header}")
                            print("Error:", e)

                        try:
                            header = 'TOUGHNESS'
                            if y_none:
                                ymin = None
                                ymax = None
                            else:
                                ymin = None
                                ymax = None
                            master_plots(df, header, ymin, ymax, savepath, experiment_name)
                        except Exception as e:
                            print(f"ERROR with {header}")
                            print("Error:", e)
                        
                        try:
                            header = 'ELASTIC GRADIENT'
                            if y_none:
                                ymin = None
                                ymax = None
                            else:
                                ymin = 0
                                ymax =  140
                            master_plots(df, header, ymin, ymax, savepath, experiment_name)
                        except Exception as e:
                            print(f"ERROR with {header}")
                            print("Error:", e)

                        try:
                            header = 'ELASTIC EMOD'
                            if y_none:
                                ymin = None
                                ymax = None
                            else:
                                ymin = 2.5
                                ymax =  6.0
                            master_plots(df, header, ymin, ymax, savepath, experiment_name)
                        except Exception as e:
                            print(f"ERROR with {header}")
                            print("Error:", e)

                        # scatter plots
                        extra_savepath = f"{savepath}extra_information/"
                        os.makedirs(extra_savepath, exist_ok=True)

                        y_col = 'ELASTIC EMOD'
                        x_col = 'MEAN DIAMETER'
                        savedir = f"{extra_savepath}correlation_plot/"
                        plt = plotting.create_scatter_plot(df, x_col, y_col, savedir)

                        y_col = 'ELASTIC EMOD'
                        x_col = 'MIN DIAMETER'
                        savedir = f"{extra_savepath}correlation_plot/"
                        plt = plotting.create_scatter_plot(df, x_col, y_col, savedir)

                        y_col = 'ELASTIC EMOD'
                        x_col = 'MAX DIAMETER'
                        savedir = f"{extra_savepath}correlation_plot/"
                        plt = plotting.create_scatter_plot(df, x_col, y_col, savedir)

                        y_col = 'BREAK STRESS'
                        x_col = 'MEAN DIAMETER'
                        savedir = f"{extra_savepath}correlation_plot/"
                        plt = plotting.create_scatter_plot(df, x_col, y_col, savedir)

                        y_col = 'BREAK STRESS'
                        x_col = 'MIN DIAMETER'
                        savedir = f"{extra_savepath}correlation_plot/"
                        plt = plotting.create_scatter_plot(df, x_col, y_col, savedir)

                        y_col = 'BREAK STRESS'
                        x_col = 'MAX DIAMETER'
                        savedir = f"{extra_savepath}correlation_plot/"
                        plt = plotting.create_scatter_plot(df, x_col, y_col, savedir)

                        y_col = 'BREAK STRESS'
                        x_col = 'RECORD'
                        savedir = f"{extra_savepath}correlation_plot/"
                        plt = plotting.create_scatter_plot(df, x_col, y_col, savedir)

                    except:
                        print(f"\nERROR with {name}\n")
                        continue


        # plotting single specific plots
        if MODE == "single" and not args.summary_only:
            print("\n\nplotting single data")
            for df, name, experiment_name in zip(all_dfs, all_fns, all_exp_names):
                try:
                    savepath = f"{root}{name}/"
                    if dev:
                        savepath = f"{root}dev/"
                    os.makedirs(savepath, exist_ok=True)
                    
                    try:
                        header = 'TENSILE_STRENGTH'
                        if y_none:
                            ymin = None
                            ymax = None
                        else:
                            ymin = 90
                            ymax = 325
                        master_plots(df, header, ymin, ymax, savepath, experiment_name)
                    except Exception as e:
                        print(f"ERROR with {header}")
                        print("Error:", e)

                    try:
                        header = 'BREAK_STRAIN(*)(#)'
                        if y_none:
                            ymin = None
                            ymax = None
                        else:
                            ymin = 0
                            ymax = 100
                        master_plots(df, header, ymin, ymax, savepath, experiment_name)
                    except Exception as e:
                        print(f"ERROR with {header}")
                        print("Error:", e)

                    try:
                        header = 'BREAK_LOAD'
                        if y_none:
                            ymin = None
                            ymax = None
                        else:
                            ymin = 0
                            ymax = 2.0
                        master_plots(df, header, ymin, ymax, savepath, experiment_name)
                    except Exception as e:
                        print(f"ERROR with {header}")
                        print("Error:", e)
                    
                    try:
                        header = 'EMOD(*)(#)'
                        if y_none:
                            ymin = None
                            ymax = None
                        else:
                            ymin = 2.5
                            ymax = 8.0
                        master_plots(df, header, ymin, ymax, savepath, experiment_name)
                    except Exception as e:
                        print(f"ERROR with {header}")
                        print("Error:", e)

                    # scatter plots
                    extra_savepath = f"{savepath}extra_information/"
                    os.makedirs(extra_savepath, exist_ok=True)

                    y_col = 'TENSILE_STRENGTH'
                    x_col = 'MEAN AREA'
                    savedir = f"{extra_savepath}correlation_plot/"
                    plt = plotting.create_scatter_plot(df, x_col, y_col, savedir)

                    y_col = 'EMOD(*)(#)'
                    x_col = 'MEAN AREA'
                    savedir = f"{extra_savepath}correlation_plot/"
                    plt = plotting.create_scatter_plot(df, x_col, y_col, savedir)

                    y_col = 'BREAK_LOAD'
                    x_col = 'MEAN AREA'
                    savedir = f"{extra_savepath}correlation_plot/"
                    plt = plotting.create_scatter_plot(df, x_col, y_col, savedir)

                except:
                    print(f"\nERROR with {name}\n")
                    continue
    if not args.summary_only:
        print("\n\nPlotting and summary DONE\n\n")
    elif args.summary_only:
        print("\n\nSummary DONE\n\n")