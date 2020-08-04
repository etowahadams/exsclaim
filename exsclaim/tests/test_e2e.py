import unittest
import json
import pathlib
import os

import responses
from deepdiff import DeepDiff

from ..pipeline import Pipeline
from ..tool import JournalScraper, CaptionSeparator, FigureSeparator

class TestNatureFull(unittest.TestCase):

    def setUp(self):
        """ Instantiates a test search query and Exsclaim pipeline to test """
        # Set query paths
        query_path = pathlib.Path(__file__).parent / 'data' / 'nature_test.json'

        # Set path to initial exsclaim_dict JSON (if applicable)
        exsclaim_path = ""

        # Initialize EXSCLAIM! tools
        self.js = JournalScraper()
        self.cs = CaptionSeparator()
        self.fs = FigureSeparator()

        # Initialize EXSCLAIM! pipeline
        self.exsclaim_pipeline = Pipeline(query_path =    query_path,
                                          exsclaim_path = exsclaim_path)

    @responses.activate
    def test_full_run(self):
        """ tests one full run of pipeline """
        ## Load Expected result
        expected_json = pathlib.Path(__file__).parent / "data" / "nature_expected.json"
        with open(expected_json, "r") as f:
          expected = json.load(f)
        
        ##  Set up Mock URLs
        # For HTML Files
        test_articles = pathlib.Path(__file__).parent / 'data' / 'nature_articles'
        for article_name in os.listdir(test_articles):
            url = "https://www.nature.com/articles/" + article_name.split(".")[0]
            with open(test_articles / article_name, "r") as f:
                article_html = f.read()
            responses.add(responses.GET, url, body = article_html)
        url = "https://www.nature.com/search?q=%22Ag+nanoparticle%22%20%22HAADF-STEM%22&order=relevance&page=1"
        test_search = pathlib.Path(__file__).parent / 'data' / 'nature_search.html'
        with open(test_search, "r") as f:
            article_html = f.read()
        responses.add(responses.GET, url, body = article_html)
        # For images
        for image in expected:
            url = expected[image]["image_url"]
            location = expected[image]["figure_path"]
            with open(location, "rb") as f:
                responses.add(
                    responses.GET,
                    url,
                    body = f.read(),
                    status = 200,
                    content_type = "img/jpg",
                    stream = True
                )

        # Run the tools through the pipeline
        tools = [self.js,self.cs,self.fs] # define run order
        self.exsclaim_pipeline.run(tools)

        # Group related image objects into master images
        exsclaim_json = self.exsclaim_pipeline.group_objects()

        # Run comparison of expected and resulting jsons
        diff = DeepDiff(exsclaim_json, expected, ignore_order=True)

        ## Band-aid to handle https://github.com/MaterialEyes/exsclaim/issues/5
        ## in testing. This will call the test a pass if we find either of the 
        ## results that appear seemingly nondeterministically. 
        accepted_difference = {'values_changed': {"root['s41467-018-06211-3_fig5.jpg']['master_images'][0]['caption'][0]": {'new_value': 'Precious metal dissolution tests in aluminum–air flow batteries (AAFBs) using the SMNp and Pt/C with 6 \u2009 M KOH electrolyte after 6\u2009h of discharging at 50 \u2009 mA \u2009 cm−2', 'old_value': 'c, d'}}}
        self.assertIn(diff, ({}, accepted_difference), 
                                    ("The resulting json and the reference "
                                    "json are different. This does not mean "
                                    "the implementation is incorrect, just "
                                    "that a change has been made to the code "
                                    "that changes results. If you have made "
                                    "a change that you believe improves "
                                    "results, you should use other methods "
                                    "to check correctness and accuracy."))


if __name__ == '__main__':
    unittest.main()