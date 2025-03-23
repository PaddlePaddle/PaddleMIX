from typing import Dict
from ...core import MMDataset, register
from functools import partial

# Define the conversion function
def convert_llava_item(item: Dict, image_path_prefix: str = '') -> Dict:
    """
    Convert each data item to the target format.
    
    Args:
        item (dict): Original data item containing 'image' and 'conversations' keys.
        image_path_prefix (str): Prefix for the image path. Defaults to an empty string. 
    
    Returns:
        dict: Transformed data item containing 'image' and 'conversations' keys.
    """

    # Check if the 'image' key exists, if not, set it to an empty string
    image = item.get('image', '')  # Default to an empty string if 'image' key is missing

    # Skip this item if the 'image' field is empty
    if not image:
        return None  # Skip this item by returning None if no image exists
    
    # Concatenate the image path
    image = image_path_prefix + image
    # print(item['conversations'])
    
    conversations = []
    for i in range(0, len(item['conversations']), 2):
        human_message = item['conversations'][i]['value']
        gpt_message = item['conversations'][i+1]['value'] if i+1 < len(item['conversations']) else ''
        conversations.append([human_message, gpt_message])

    # Construct the transformed data structure
    transformed_item = {
        "image": image,
        "conversations": conversations
    }

    return transformed_item

@register()
def llava_convert(dataset: MMDataset, image_path_prefix='') -> MMDataset:

    print('Converting llava dataset...')
    # Use the map operator for batch transformation
    filter_func = partial(convert_llava_item, image_path_prefix=image_path_prefix)

    # Apply dataset.map
    dataset = dataset.map(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return dataset