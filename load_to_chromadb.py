import os
from dotenv import load_dotenv
from IPython.display import Markdown, display

from llama_index.legacy import VectorStoreIndex, ServiceContext
from llama_index.legacy.vector_stores import ChromaVectorStore
from llama_index.legacy.storage.storage_context import StorageContext
from llama_index.legacy.embeddings import HuggingFaceEmbedding
from llama_index.legacy.llms import Gemini
from llama_index.legacy.node_parser import SentenceWindowNodeParser, SimpleNodeParser
from llama_index.legacy.llms import Gemini
from llama_index.legacy import GPTVectorStoreIndex
from llama_index.legacy.readers.web import BeautifulSoupWebReader

import chromadb
import streamlit as st

# Enable Logging
import logging
import sys

#You can set the logging level to DEBUG for more verbose output,
# or use level=logging.INFO for less detailed information.
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# Load environment variables from the .env file
load_dotenv()

loader = BeautifulSoupWebReader()

urls = [
    'https://www.hsph.harvard.edu/nutritionsource/kids-healthy-eating-plate/',
    'https://www.hsph.harvard.edu/nutritionsource/healthy-eating-plate/',
    'https://www.hsph.harvard.edu/nutritionsource/what-should-you-eat/',
    'https://www.hsph.harvard.edu/nutritionsource/what-should-you-eat/whole-grains/',
    'https://www.hsph.harvard.edu/nutritionsource/what-should-you-eat/protein/',
    'https://www.hsph.harvard.edu/nutritionsource/what-should-you-eat/vegetables-and-fruits/',
    'https://www.hsph.harvard.edu/nutritionsource/what-should-you-eat/fats-and-cholesterol/',
    'https://www.hsph.harvard.edu/nutritionsource/what-should-you-eat/fats-and-cholesterol/types-of-fat/',
    'https://www.hsph.harvard.edu/nutritionsource/what-should-you-eat/fats-and-cholesterol/cholesterol/',
    'https://www.hsph.harvard.edu/nutritionsource/what-should-you-eat/fats-and-cholesterol/dietary-fat-and-disease/',
    'https://www.hsph.harvard.edu/nutritionsource/vitamins/',
    'https://www.hsph.harvard.edu/nutritionsource/healthy-drinks/',
    'https://www.hsph.harvard.edu/nutritionsource/healthy-drinks/other-healthy-beverage-options/', 
    'https://www.hsph.harvard.edu/nutritionsource/healthy-drinks/drinks-to-consume-in-moderation/', 
    'https://www.hsph.harvard.edu/nutritionsource/healthy-drinks/sugary-drinks/', 
    'https://www.hsph.harvard.edu/nutritionsource/sports-drinks/', 
    'https://www.hsph.harvard.edu/nutritionsource/energy-drinks/', 
    'https://www.hsph.harvard.edu/nutritionsource/healthy-drinks/beverages-public-health-concerns/', 
    'https://www.hsph.harvard.edu/nutritionsource/healthy-drinks/artificial-sweeteners/', 
    'https://www.hsph.harvard.edu/nutritionsource/salt-and-sodium/', 
    'https://www.hsph.harvard.edu/nutritionsource/salt-and-sodium/take-action-on-salt/', 
    'https://www.hsph.harvard.edu/nutritionsource/salt-and-sodium/sodium-public-health-concerns/', 
    'https://www.hsph.harvard.edu/nutritionsource/carbohydrates/', 
    'https://www.hsph.harvard.edu/nutritionsource/carbohydrates/carbohydrates-and-blood-sugar/', 
    'https://www.hsph.harvard.edu/nutritionsource/carbohydrates/fiber/', 
    'https://www.hsph.harvard.edu/nutritionsource/carbohydrates/added-sugar-in-the-diet/', 
    'https://www.hsph.harvard.edu/nutritionsource/sustainability/', 
    'https://www.hsph.harvard.edu/nutritionsource/sustainability/plate-and-planet/', 
    'https://www.hsph.harvard.edu/nutritionsource/sustainability/food-waste/', 
    'https://www.hsph.harvard.edu/nutritionsource/healthy-weight/', 
    'https://www.hsph.harvard.edu/nutritionsource/healthy-weight/measuring-fat/', 
    'https://www.hsph.harvard.edu/nutritionsource/healthy-weight/best-diet-quality-counts/', 
    'https://www.hsph.harvard.edu/nutritionsource/healthy-weight/healthy-dietary-styles/', 
    'https://www.hsph.harvard.edu/nutritionsource/healthy-weight/diet-reviews/', 
    'https://www.hsph.harvard.edu/nutritionsource/staying-active/', 
    'https://www.hsph.harvard.edu/nutritionsource/staying-active/active-communities/', 
    'https://www.hsph.harvard.edu/nutritionsource/stress-and-health/', 
    'https://www.hsph.harvard.edu/nutritionsource/sleep/', 
    'https://www.hsph.harvard.edu/nutritionsource/healthy-longevity/', 
    'https://www.hsph.harvard.edu/nutritionsource/disease-prevention/', 
    'https://www.hsph.harvard.edu/nutritionsource/disease-prevention/cardiovascular-disease/', 
    'https://www.hsph.harvard.edu/nutritionsource/disease-prevention/cardiovascular-disease/preventing-cvd/', 
    'https://www.hsph.harvard.edu/nutritionsource/disease-prevention/diabetes-prevention/', 
    'https://www.hsph.harvard.edu/nutritionsource/disease-prevention/diabetes-prevention/preventing-diabetes-full-story/', 
    'https://www.hsph.harvard.edu/nutritionsource/cancer/', 
    'https://www.hsph.harvard.edu/nutritionsource/cancer/preventing-cancer/', 
    'https://www.hsph.harvard.edu/nutritionsource/oral-health/', 
    'https://www.hsph.harvard.edu/nutritionsource/precision-nutrition/', 
    'https://www.hsph.harvard.edu/nutritionsource/nutrition-and-immunity/', 
    'https://www.hsph.harvard.edu/nutritionsource/recipes-2/', 
    'https://www.hsph.harvard.edu/nutritionsource/asparagus-with-warm-tarragon-pecan-vinaigrette/', 
    'https://www.hsph.harvard.edu/nutritionsource/asparagus-spears-with-mandarin-orange/', 
    'https://www.hsph.harvard.edu/nutritionsource/baby-arugula-and-shaved-fennel-with-lemon-vinaigrette/',
    'https://www.hsph.harvard.edu/nutritionsource/braised-cabbage-with-leeks-and-sesame-seeds/',
    'https://www.hsph.harvard.edu/nutritionsource/braised-oyster-mushrooms-coconut-macadamia/',
    'https://www.hsph.harvard.edu/nutritionsource/butternut-squash-soup-recipe/',
    'https://www.hsph.harvard.edu/nutritionsource/caesar-salad/',
    'https://www.hsph.harvard.edu/nutritionsource/cardamom-roasted-cauliflower/',
    'https://www.hsph.harvard.edu/nutritionsource/carrot-and-coriander-soup/',
    'https://www.hsph.harvard.edu/nutritionsource/cauliflower-tomato-soup/',
    'https://www.hsph.harvard.edu/nutritionsource/cauliflower-walnut-soup/',
    'https://www.hsph.harvard.edu/nutritionsource/endive-salad-with-citrus-walnut-dressing/',
    'https://www.hsph.harvard.edu/nutritionsource/customizable-stuffed-peppers/',
    'https://www.hsph.harvard.edu/nutritionsource/fresh-spinach-with-sesame-seeds/',
    'https://www.hsph.harvard.edu/nutritionsource/garlic-braised-greens/',
    'https://www.hsph.harvard.edu/nutritionsource/green-beans-with-dried-cherries/',
    'https://www.hsph.harvard.edu/nutritionsource/green-beans-with-chili-garlic-sauce/',
    'https://www.hsph.harvard.edu/nutritionsource/green-chutney/',
    'https://www.hsph.harvard.edu/nutritionsource/grilled-eggplant-cutlets/',
    'https://www.hsph.harvard.edu/nutritionsource/kale-with-caramelized-onions/',
    'https://www.hsph.harvard.edu/nutritionsource/marinated-shiitake-mushroom-and-cucumber-salad/',
    'https://www.hsph.harvard.edu/nutritionsource/mashed-cauliflower/',
    'https://www.hsph.harvard.edu/nutritionsource/mushroom-stroganoff/',
    'https://www.hsph.harvard.edu/nutritionsource/pan-roasted-wild-mushrooms-with-coffee-and-hazelnuts/',
    'https://www.hsph.harvard.edu/nutritionsource/portabella-steak-sandwich/',
    'https://www.hsph.harvard.edu/nutritionsource/provencal-vegetables/',
    'https://www.hsph.harvard.edu/nutritionsource/vegetable-stock/',
    'https://www.hsph.harvard.edu/nutritionsource/roasted-brussels-sprouts/',
    'https://www.hsph.harvard.edu/nutritionsource/brussels-sprouts-with-shallots/',
    'https://www.hsph.harvard.edu/nutritionsource/roasted-beets-with-balsamic-vinegar/',
    'https://www.hsph.harvard.edu/nutritionsource/roasted-balsamic-vegetables/',
    'https://www.hsph.harvard.edu/nutritionsource/roasted-squash-with-pomegranate/',
    'https://www.hsph.harvard.edu/nutritionsource/sweet-potatoes-with-pecans/',
    'https://www.hsph.harvard.edu/nutritionsource/ruby-chard/',
    'https://www.hsph.harvard.edu/nutritionsource/sauted-rainbow-swiss-chard/',
    'https://www.hsph.harvard.edu/nutritionsource/simple-celery-date-salad/',
    'https://www.hsph.harvard.edu/nutritionsource/southwestern-corn-hash/',
    'https://www.hsph.harvard.edu/nutritionsource/spicy-broccolini/',
    'https://www.hsph.harvard.edu/nutritionsource/spicy-indian-slaw/',
    'https://www.hsph.harvard.edu/nutritionsource/stir-fried-vegetables-tomato-curry/',
    'https://www.hsph.harvard.edu/nutritionsource/sugar-snap-peas-with-fresh-mint/',
    'https://www.hsph.harvard.edu/nutritionsource/tarragon-succotash/',
    'https://www.hsph.harvard.edu/nutritionsource/tunisian-carrot-salad/',
    'https://www.hsph.harvard.edu/nutritionsource/vegetable-stock-recipe/',
    'https://www.hsph.harvard.edu/nutritionsource/vegetarian-shepherds-pie-recipe/',
    'https://www.hsph.harvard.edu/nutritionsource/wild-mushroom-soup-with-soba/',
    'https://www.hsph.harvard.edu/nutritionsource/yellow-squash-with-sage/',
    'https://www.hsph.harvard.edu/nutritionsource/arugula-watermelon-feta-and-mint-salad-with-balsamic-vinaigrette/',
    'https://www.hsph.harvard.edu/nutritionsource/citrus-salad/',
    'https://www.hsph.harvard.edu/nutritionsource/almond-coconut-macaroons/',
    'https://www.hsph.harvard.edu/nutritionsource/dried-fruit-and-nuts/',
    'https://www.hsph.harvard.edu/nutritionsource/watermelon-salad/',
    'https://www.hsph.harvard.edu/nutritionsource/fruit-compote-spiced-nuts/',
    'https://www.hsph.harvard.edu/nutritionsource/strawberry-rhubarb-crisp/',
    'https://www.hsph.harvard.edu/nutritionsource/barley-roasted-portobello-and-fennel-salad/',
    'https://www.hsph.harvard.edu/nutritionsource/blueberry-muffins/',
    'https://www.hsph.harvard.edu/nutritionsource/brown-rice-pancakes/',
    'https://www.hsph.harvard.edu/nutritionsource/bulgur-pilaf/',
    'https://www.hsph.harvard.edu/nutritionsource/couscous-minted-with-pine-nuts/',
    'https://www.hsph.harvard.edu/nutritionsource/couscous-quinoa-tabouli/',
    'https://www.hsph.harvard.edu/nutritionsource/cranberry-orange-muffin/',
    'https://www.hsph.harvard.edu/nutritionsource/fantastic-bulgur-dish/',
    'https://www.hsph.harvard.edu/nutritionsource/farro-risotto-walnut-pesto/',
    'https://www.hsph.harvard.edu/nutritionsource/farro-roasted-confetti-vegetables/',
    'https://www.hsph.harvard.edu/nutritionsource/hearty-whole-grain-bread/',
    'https://www.hsph.harvard.edu/nutritionsource/irish-brown-bread/',
    'https://www.hsph.harvard.edu/nutritionsource/jalapeno-cheddar-corn-muffins/',
    'https://www.hsph.harvard.edu/nutritionsource/lemon-chickpea-breakfast-muffins/',
    'https://www.hsph.harvard.edu/nutritionsource/mediterranean-rice/',
    'https://www.hsph.harvard.edu/nutritionsource/mixed-up-grains/',
    'https://www.hsph.harvard.edu/nutritionsource/mushroom-barley-risotto/',
    'https://www.hsph.harvard.edu/nutritionsource/oatmeal-roti/',
    'https://www.hsph.harvard.edu/nutritionsource/pasta-in-zemino/',
    'https://www.hsph.harvard.edu/nutritionsource/rigatoni-fresh-basil-pesto-corn-zucchini/',
    'https://www.hsph.harvard.edu/nutritionsource/quinoa-chia-edamame-veggie-burger/',
    'https://www.hsph.harvard.edu/nutritionsource/quinoa-enchilada-casserole/',
    'https://www.hsph.harvard.edu/nutritionsource/spicy-coconut-rice-with-limes/',
    'https://www.hsph.harvard.edu/nutritionsource/three-green-wheat-berry-salad-with-mushroom-bacon-recipe/',
    'https://www.hsph.harvard.edu/nutritionsource/wheatberries-and-chives/',
    'https://www.hsph.harvard.edu/nutritionsource/whole-wheat-banana-nut-muffins/',
    'https://www.hsph.harvard.edu/nutritionsource/whole-wheat-penne-with-pistachio-pesto-and-cherry-tomatoes/',
    'https://www.hsph.harvard.edu/nutritionsource/wild-rice-with-cranberries/',
    'https://www.hsph.harvard.edu/nutritionsource/greek-skordalia/',
    'https://www.hsph.harvard.edu/nutritionsource/green-lentil-hummus-herbs-olives/',
    'https://www.hsph.harvard.edu/nutritionsource/guacamole/',
    'https://www.hsph.harvard.edu/nutritionsource/hot-pepper-vinaigrette/',
    'https://www.hsph.harvard.edu/nutritionsource/hummus/',
    'https://www.hsph.harvard.edu/nutritionsource/italian-pesto-alla-trapanese/',
    'https://www.hsph.harvard.edu/nutritionsource/mint-vinaigrette/',
    'https://www.hsph.harvard.edu/nutritionsource/oregano-garlic-vinaigrette/',
    'https://www.hsph.harvard.edu/nutritionsource/spanish-romesco-sauce/',
    'https://www.hsph.harvard.edu/nutritionsource/turkish-muhammara/',
    'https://www.hsph.harvard.edu/nutritionsource/turkish-tarator/',
    'https://www.hsph.harvard.edu/nutritionsource/walnut-pesto/',
    'https://www.hsph.harvard.edu/nutritionsource/white-bean-and-kale-hummus/',
    'https://www.hsph.harvard.edu/nutritionsource/asian-trail-mix/',
    'https://www.hsph.harvard.edu/nutritionsource/cozy-red-lentil-mash/',
    'https://www.hsph.harvard.edu/nutritionsource/crunchy-roasted-chickpeas/',
    'https://www.hsph.harvard.edu/nutritionsource/curried-red-lentil-soup/',
    'https://www.hsph.harvard.edu/nutritionsource/dukkah/',
    'https://www.hsph.harvard.edu/nutritionsource/french-style-lentils/',
    'https://www.hsph.harvard.edu/nutritionsource/garbanzo-beans-with-spinach-and-tomatoes/',
    'https://www.hsph.harvard.edu/nutritionsource/green-beans-with-tofu-and-crushed-peanuts/',
    'https://www.hsph.harvard.edu/nutritionsource/mushroom-tofu-veggie-burger/',
    'https://www.hsph.harvard.edu/nutritionsource/spicy-lemongrass-tofu-with-asian-basil/',
    'https://www.hsph.harvard.edu/nutritionsource/sprouted-lentil-cabbage-celery-slaw/',
    'https://www.hsph.harvard.edu/nutritionsource/thai-eggplant-salad-with-coconut-tofu-strips/',
    'https://www.hsph.harvard.edu/nutritionsource/tomato-and-white-bean-salad/',
    'https://www.hsph.harvard.edu/nutritionsource/whole-wheat-penne-with-pistachio-pesto-and-cherry-tomatoes/',
    'https://www.hsph.harvard.edu/nutritionsource/white-beans-wild-rice-and-mushrooms/',
    'https://www.hsph.harvard.edu/nutritionsource/vegetarian-refried-beans/',
    'https://www.hsph.harvard.edu/nutritionsource/cod-and-littleneck-clams/',
    'https://www.hsph.harvard.edu/nutritionsource/crawfish-touffe/',
    'https://www.hsph.harvard.edu/nutritionsource/crispy-pan-seared-white-fish-walnut-romesco-pea-shoot-salad/',
    'https://www.hsph.harvard.edu/nutritionsource/fish-creole/',
    'https://www.hsph.harvard.edu/nutritionsource/miso-marinated-salmon-grilled-alder-wood/',
    'https://www.hsph.harvard.edu/nutritionsource/pan-roasted-salmon-with-dill-olive-oil-capers/',
    'https://www.hsph.harvard.edu/nutritionsource/pan-roasted-salmon/',
    'https://www.hsph.harvard.edu/nutritionsource/shaved-fennel-salad-coriander-crusted-hamachi/',
    'https://www.hsph.harvard.edu/nutritionsource/shrimp-and-chicken-gumbo/',
    'https://www.hsph.harvard.edu/nutritionsource/shrimp-red-curry-crispy-sprouted-lentils/',
    'https://www.hsph.harvard.edu/nutritionsource/wild-salmon-salad/',
    'https://www.hsph.harvard.edu/nutritionsource/fish-tacos-with-cilantro-slaw/',
    'https://www.hsph.harvard.edu/nutritionsource/chicken-shrimp-and-fruit-salad/',
    'https://www.hsph.harvard.edu/nutritionsource/lemongrass-marinated-chicken-breast/',
    'https://www.hsph.harvard.edu/nutritionsource/olive-oil-dressing-with-chicken-walnuts-recipe/',
    'https://www.hsph.harvard.edu/nutritionsource/rosemary-and-lemon-grilled-chicken-breast/',
    'https://www.hsph.harvard.edu/nutritionsource/spicy-chicken-kebabs-with-moorish-flavors/',
    'https://www.hsph.harvard.edu/nutritionsource/stir-fried-chicken/',
    'https://www.hsph.harvard.edu/nutritionsource/moroccan-chicken-stew-with-apricots/',
    'https://www.hsph.harvard.edu/nutritionsource/stir-fried-chicken/',
    'https://www.hsph.harvard.edu/nutritionsource/baked-ricotta/',
    'https://www.hsph.harvard.edu/nutritionsource/roasted-tomatoes-stuffed-goat-cheese-garlic-basil/',
    'https://www.hsph.harvard.edu/nutritionsource/fruit-cooler/',
    'https://www.hsph.harvard.edu/nutritionsource/iced-tea-with-lemon-and-mint/'
    
    # Add the rest of the URLs here   
]

documents = loader.load_data(urls=urls)


# base Query Engine LLM
llm = Gemini(api_key=os.getenv("google_api_key"),model='gemini-pro')

# fine-tuned Embeddings model
embed_model = HuggingFaceEmbedding(
    model_name='Revankumar/fine_tuned_embeddings_for_healthy_recipes'
)


# fine-tuned ServiceContext
ctx = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

parser = SimpleNodeParser()

nodes = parser.get_nodes_from_documents(documents)


db = chromadb.PersistentClient(path="./chroma_db")

chroma_collection = db.get_or_create_collection("quickstart")


vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

service_context = ServiceContext.from_defaults(embed_model=embed_model,llm=llm)

VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, service_context=service_context
)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






