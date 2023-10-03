import os.path
import sys

import streamlit as st
import pandas as pd
import random


if "i" not in st.session_state:
    st.session_state.i = 0


def generate_random_colors(length):
    colors = []
    for _ in range(length):
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        colors.append(f"rgb({red}, {green}, {blue})")
    return colors


def generate_colors_map():
    df = st.session_state['df']
    if st.session_state["cover_score"] == "document":
        representatives = df[df["rank"] <= st.session_state['num_clusters']].representative.unique()
        # sort representatives by rank
        representatives = sorted(representatives, key=lambda x: df[df["representative"] == x]["rank"].iloc[0])
    else:
        representatives = df[df[f"best_{st.session_state['num_clusters']}"] == True].representative.unique()
    colors = generate_random_colors(len(representatives))
    color_map = {}
    for i, rep in enumerate(representatives):
        color_map[rep] = colors[i]
    st.session_state['color_map'] = color_map


def generate_sidebar_linking(color_map, line_numbers):
    for representative in color_map:
        id_rep = get_id_rep(representative)
        if representative in line_numbers:
            st.sidebar.markdown(
                f"<a style='border: 3px solid {color_map[representative]}; padding: 5px; font-size: 16px; color: black;' href='#{id_rep}'>{representative}</a>",
                unsafe_allow_html=True)
        else:
            st.sidebar.markdown(
                f"<a style='border: 3px solid {color_map[representative]}; padding: 5px; font-size: 16px; color: gray;'>{representative}</a>",
                unsafe_allow_html=True)


@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)


def load_new_csv():
    dir_name = os.path.join(os.path.dirname(__file__), "output/CTOC/CUAD/1002/")
    weights_mapping = {"title:1": "w-title-1",
                       "title:0.5, index:0.5": "w-title-05-w-index-05",
                       "title:0.5, index:0.3, body:0.2": "w-title-05-w-index-03-w-text-02",
                       "title:0.33, index:0.33, body:0.33": "equal-similarity-w"}
    full_dir_name = dir_name + weights_mapping[st.session_state["weights"]]
    with_model = os.path.join(full_dir_name, st.session_state["model_name"])
    csv_file = os.path.join(with_model, "meta_filtered.csv")
    st.session_state['df'] = load_csv(csv_file)
    generate_colors_map()


def main():
    # Page title and description
    st.title("Conceptual ToC Viewer")

    st.write("Select which version of the ToC you want to view.")

    st.selectbox("model name",
                 ["all-roberta-large-v1", "all-mpnet-base-v2", "gtr-t5-large"],
                 key="model_name",
                 on_change=load_new_csv)

    st.selectbox("weights",
                 ["title:1", "title:0.5, index:0.5", "title:0.5, index:0.3, body:0.2", "title:0.33, index:0.33, body:0.33"],
                 key="weights",
                 on_change=load_new_csv)

    st.selectbox("cover score",
                 ["document", "collection"],
                 key="cover_score",
                 on_change=generate_colors_map)

    st.number_input("num clusters to display", min_value=1,
                    max_value=30 if st.session_state["cover_score"] == "document" else 15,
                    key="num_clusters",
                    on_change=generate_colors_map)



    # Load CSV data
    if 'df' not in st.session_state:
        load_new_csv()

    df = st.session_state['df']
    color_map = st.session_state['color_map']

    st.sidebar.markdown("<h3 style='font-size: 24px;'>ToC (color mapping)</h3>",
                        unsafe_allow_html=True)

    group_by_filename = df.groupby("filename").groups

    # Display file selection dropdown
    selected_file = st.selectbox("Select a file", group_by_filename.keys())

    # Filter dataframe based on selected file
    filtered_df = df.loc[group_by_filename[selected_file]]
    filtered_df = filtered_df.sort_values(by=['title_index']).reset_index()

    if not filtered_df.empty:
        display_single_file(color_map, filtered_df)
    else:
        st.write("Selected file not found in the CSV.")


def display_single_file(color_map, filtered_df):

    st.header("Text Content:")

    all_paragraphs, labels_start_end = get_paragraphs(filtered_df)

    line_numbers = {}
    is_open = (False, None)
    for i, paragraph in enumerate(all_paragraphs):
        prev_label, current_label = labels_start_end[i]
        if prev_label is not None:
            line_numbers[prev_label] = (line_numbers[prev_label], i)
            color = color_map[prev_label]
            st.markdown(f"""<hr style="height:10px;border:none;color:{color};background-color:{color};" /> """, unsafe_allow_html=True)
            is_open = (False, None)
        if current_label is not None:
            id_rep = get_id_rep(current_label)
            st.markdown(f"<h3 id='{id_rep}'>{current_label}</h3>", unsafe_allow_html=True)  # todo add id for the hyperlink, write this as a header
            color = color_map[current_label]
            st.markdown(f"""<hr style="height:10px;border:none;color:{color};background-color:{color};" /> """,
                        unsafe_allow_html=True)
            is_open = (True, current_label)
            line_numbers[current_label] = i+1

        st.markdown(paragraph)

    if is_open[0]:
        label = is_open[1]
        color = color_map[label]
        st.markdown(
            f"""<hr style="height:10px;border:none;color:{color};background-color:{color};" /> """,
            unsafe_allow_html=True)
        line_numbers[label] = (line_numbers[label], len(all_paragraphs))

    generate_sidebar_linking(color_map, line_numbers)


def get_id_rep(representative):
    return representative.lower().replace('.', '').replace(' ', '-')


def get_paragraphs(filtered_df):
    all_paragraphs = []
    all_labels = []
    labels_start_end = []
    for i, row in filtered_df.iterrows():
        paragraph = f"{row['title_text']}\n\n{row['section_text']}\n\n"
        if i + 1 < len(filtered_df):
            if row['section_text'] == filtered_df.loc[i + 1]['title_text']:
                paragraph += f"{filtered_df.loc[i + 1]['section_text']}\n\n"
        label = row["representative"] if row["rank"] <= st.session_state['num_clusters'] else None
        prev = current = None
        if i == 0 and label is not None:
            current = label
        if i > 0 and all_labels[-1] != label:
            if i > 0 and all_labels[-1] is not None:
                prev = all_labels[-1]
            if label is not None:
                current = label
        labels_start_end.append((prev, current))
        all_paragraphs.append(paragraph)
        all_labels.append(label)

    return all_paragraphs, labels_start_end


if __name__ == '__main__':
    main()
