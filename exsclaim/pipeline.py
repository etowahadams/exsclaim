# In productivity.py
import os
import csv
import utils
import json
import time
import cluster
import matplotlib.pyplot as plt

class Pipeline:
    def __init__(self , query_path, exsclaim_path):
        self.query_path = query_path
        self.exsclaim_path = exsclaim_path
        try:
            with open(exsclaim_path, 'r') as f:
                # Load configuration file values
                self.exsclaim_dict = json.load(f)
        except FileNotFoundError:
            # Keep preset values
            self.exsclaim_dict = {}

    def _load_query(self):
    	with open(self.query_path) as f:
    		return(json.load(f))

    def run(self, tools):
        print("""
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@&   /&@@@(   /@@@@@@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@ %@@@@@@@@@@@@@@@@@@@ *@@@@@@@@@@@@@@
        @@@@@@@@@@@@ @@@@@@@@@@@@@@,  .@@@@@@@@ *@@@@@@@@@@@
        @@@@@@@@@.#@@@@@@@@@@@@@@@@,    @@@@@@@@@@ @@@@@@@@@
        @@@@@@@&,@@@@@@@@@@@@@@@@@@.    @@@@@@@@@@@@ @@@@@@@
        @@@@@@ @@@@@@@@@@@@@@@@@@@@     @@@@@@@@@@@@@ @@@@@@
        @@@@@ @@@@@@@@@@@@@@@@@@@@@    *@@@@@@@@@@@@@@/@@@@@
        @@@@ @@@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@@@,@@@@
        @@@ @@@@@@@@@@@@@@@@@@@@@@&    @@@@@@@@@@@@@@@@@ @@@
        @@@,@@@@@@@@@@@@@@@@@@@@@@*   (@@@@@@@@@@@@@@@@@@%@@
        @@.@@@@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@@@@@@ @@
        @@ @@@@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@@@@@@ @@
        @@ @@@@@@@@@@@@@@@@@@@@@@/   &@@@@@@@@@@@@@@@@@@@ @@
        @@,@@@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@@@@@@@ @@
        @@@.@@@@@@@@@@@@@@@@@@@@&   @@@@@@@@@@@@@@@@@@@@@%@@
        @@@ @@@@@@@@@@@@@@@@@@@@@  /@@@@@@@@@@@@@@@@@@@@ @@@
        @@@@ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@,@@@@
        @@@@@ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*@@@@@
        @@@@@@ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @@@@@@
        @@@@@@@@ @@@@@@@@@@@@    @@@@@@@@@@@@@@@@@@@ @@@@@@@
        @@@@@@@@@.(@@@@@@@@@@     @@@@@@@@@@@@@@@@ @@@@@@@@@
        @@@@@@@@@@@@ @@@@@@@@@#   #@@@@@@@@@@@@ /@@@@@@@@@@@
        @@@@@@@@@@@@@@@ ,@@@@@@@@@@@@@@@@@@@ &@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@   ,%@@&/   (@@@@@@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        """)
        search_query = self._load_query()
        for tool in tools:
            self.exsclaim_dict = tool.run(search_query,self.exsclaim_dict)
        return self.exsclaim_dict

    def group_objects(self):
        """
        Gather image objects that are part of the "unassigned" exsclaim_dict entry 
        and group together based on their association with a given subfigure label.
        """
        search_query = self._load_query()
        utils.Printer("Matching Image Objects to Caption Text\n")
        counter = 1
        for figure in self.exsclaim_dict:
            utils.Printer(">>> ({0} of {1}) ".format(counter,+\
                len(self.exsclaim_dict))+\
                "Matching objects from figure: "+figure)
    
            figure_json = self.exsclaim_dict[figure]
            masters, unassigned = cluster.assign_subfigure_labels(figure_json)
            figure_json["master_images"] = masters
            figure_json["unassigned"] = unassigned

            masters, unassigned = cluster.assign_inset_images(figure_json)
            figure_json["master_images"] = masters
            figure_json["unassigned"] = unassigned

            masters, unassigned = cluster.assign_dependent_images(figure_json)
            figure_json["master_images"] = masters
            figure_json["unassigned"] = unassigned

            scale_bars, unassigned = cluster.make_scale_bars(figure_json) 
            figure_json["unassigned"] = unassigned

            masters, unassigned = cluster.assign_scale_bars(figure_json, scale_bars)

            figure_json["master_images"] = masters
            figure_json["unassigned"] = unassigned

            masters, unassigned = cluster.assign_captions(figure_json)
            figure_json["master_images"] = masters
            figure_json["unassigned"] = unassigned

            counter +=1 
        utils.Printer(">>> SUCCESS!\n")

        with open(search_query['results_dir']+'exsclaim.json', 'w') as f:
            json.dump(self.exsclaim_dict, f, indent=3)
        return self.exsclaim_dict

    def to_file(self):
        """



        """
        search_query = self._load_query()
        utils.Printer("".join(["Printing Master Image Objects to: ",search_query['results_dir'].strip("/"),"/images","\n"]))
        rows = [['article_url', 'figure_path','figure_num', 'image_path', \
                 'master_label', 'dependent_id', 'inset_id', 'class', 'caption', 'keywords', 'scale_bar', 'pixel_size']]

        for figure_name in self.exsclaim_dict:
            fig_base, fig_ext = os.path.splitext(figure_name)
            figure = plt.imread(search_query['results_dir'] + "figures/" + figure_name)
            figure_dict = self.exsclaim_dict[figure_name]
            # Write masters to file
            for midx, mimage in enumerate(figure_dict.get("master_images", [])):
                mbase = "/".join([search_query['results_dir'].strip("/"),"images",fig_base,mimage['label']['text']])
                mcls  = 'uas' if mimage['classification'] is None else mimage['classification'][0:3].lower()
                mname = "/"+"_".join([figure_dict['figure_name'].split(fig_ext)[0],mimage['label']['text'],mcls])+fig_ext
                mpatch = utils.labelbox_to_patch(mimage['geometry'],figure)
                os.makedirs(mbase, exist_ok=True)
                mpatch =mpatch.copy(order='C')
                plt.imsave(mbase+mname,mpatch)

                rows.append([figure_dict["article_url"],figure_dict["figure_path"],fig_base.split("_fig")[-1],mbase+mname,\
                       mimage['label']['text'], None, None,\
                       mimage['classification'],mimage['caption'],mimage['keywords'],\
                       mimage.get('scale_bar', {}).get('label', {}).get('text', None),None])
                # Write dependents of masters to file
                for didx, dimage in enumerate(mimage.get("dependent_images", [])):
                    dbase = "/".join([mbase,"dependent"])
                    dcls  = 'uas' if dimage['classification'] is None else dimage['classification'][0:3].lower()
                    dname = "_".join([mname.split('par')[0]+"dep"+str(didx),dcls])+fig_ext
                    dpatch = utils.labelbox_to_patch(dimage['geometry'],figure)
                    os.makedirs(dbase, exist_ok=True)
                    plt.imsave(dbase+dname,dpatch)
    
                    rows.append([figure_dict["article_url"],figure_dict["figure_path"],fig_base.split("_fig")[-1],dbase+dname,\
                        mimage['label']['text'], str(didx), None,\
                        dimage['classification'],None, None,\
                        dimage.get('scale_bar', {}).get('label', {}).get('text', None),None])
                    # Write insets of dependents to file
                    for iidx, iimage in enumerate(dimage.get("inset_images", [])):
                        ibase = "/".join([dbase,"inset"])
                        icls  = 'uas' if iimage['classification'] is None else iimage['classification'][0:3].lower()
                        iname = "_".join([dname.split(fig_ext)[0][0:-3]+"ins"+str(iidx),icls])+fig_ext
                        
                        ipatch = utils.labelbox_to_patch(iimage['geometry'],figure)
                        os.makedirs(ibase, exist_ok=True)
                        plt.imsave(ibase+iname,ipatch)
        
                        rows.append([figure_dict["article_url"],figure_dict["figure_path"],fig_base.split("_fig")[-1],ibase+iname,\
                            mimage['label']['text'], str(didx), str(iidx),\
                            iimage['classification'],None, None,\
                            iimage.get('scale_bar', {}).get('label', {}).get('text', None),None])
                # Write insets of masters to file
                for iidx, iimage in enumerate(mimage.get("inset_images", [])):
                    ibase = "/".join([mbase,"inset"])
                    icls  = 'uas' if iimage['classification'] is None else iimage['classification'][0:3].lower()
                    iname = "_".join([mname.split(fig_ext)[0][0:-3]+"ins"+str(iidx),icls])+fig_ext
                    ipatch = utils.labelbox_to_patch(iimage['geometry'],figure)
                    os.makedirs(ibase, exist_ok=True)
                    plt.imsave(ibase+iname,ipatch)
    
                    rows.append([figure_dict["article_url"],figure_dict["figure_path"],fig_base.split("_fig")[-1],ibase+iname,\
                        mimage['label']['text'], None, str(iidx),\
                        iimage['classification'],None, None,\
                        iimage.get('scale_bar', {}).get('label', {}).get('text', None),None])

        with open(search_query['results_dir']+'labels.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(rows)
        csvFile.close()

        utils.Printer(">>> SUCCESS!\n")
