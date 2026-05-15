import pandas as pd
import numpy as np
import warnings
import streamlit as st

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.keras.utils.set_random_seed(42)   # ensures reproducible weight init across all pages/runs
np.random.seed(42)
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

from scikeras.wrappers import KerasRegressor, KerasClassifier
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Multiply, Reshape, Conv1D, Flatten, LSTM, SimpleRNN
from tensorflow.keras.optimizers import Adam

 
@st.cache_data
def load_and_merge_data():
    """Loads and merges the dataset."""
    d1 = pd.read_csv("student-mat.csv", sep=";")
    d2 = pd.read_csv("student-por.csv", sep=";")

    merged_data = pd.merge(
        d1,
        d2,
        on=[
            "school","sex","age","address","famsize","Pstatus",
            "Medu","Fedu","Mjob","Fjob","reason","nursery","internet"
        ],
        suffixes=("_mat", "_por")
    )
    return merged_data

# ===============================
# REGRESSION MODELS DEFINITION
# ===============================
def mlp_model(**kw):
    n = kw["meta"]["n_features_in_"]
    model = Sequential([
        Input(shape=(n,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(), loss="mse")
    return model

def autoencoder_reg(**kw):
    n = kw["meta"]["n_features_in_"]
    inp = Input(shape=(n,))
    encoded = Dense(64, activation="relu")(inp)
    encoded = Dense(32, activation="relu")(encoded)
    decoded = Dense(64, activation="relu")(encoded)
    out = Dense(1)(decoded)
    model = Model(inp, out)
    model.compile(optimizer=Adam(), loss="mse")
    return model

def denoising_autoencoder(**kw):
    n = kw["meta"]["n_features_in_"]
    inp = Input(shape=(n,))
    noisy = Dropout(0.2)(inp)
    encoded = Dense(64, activation="relu")(noisy)
    out = Dense(1)(encoded)
    model = Model(inp, out)
    model.compile(optimizer=Adam(), loss="mse")
    return model

def wide_deep(**kw):
    n = kw["meta"]["n_features_in_"]
    inp = Input(shape=(n,))
    wide = Dense(1)(inp)
    deep = Dense(64, activation="relu")(inp)
    deep = Dense(32, activation="relu")(deep)
    merged = wide + Dense(1)(deep)
    model = Model(inp, merged)
    model.compile(optimizer=Adam(), loss="mse")
    return model

def attention_nn(**kw):
    n = kw["meta"]["n_features_in_"]
    inp = Input(shape=(n,))
    dense = Dense(n, activation="relu")(inp)
    attn = Dense(n, activation="sigmoid")(dense)
    weighted = Multiply()([inp, attn])
    out = Dense(1)(weighted)
    model = Model(inp, out)
    model.compile(optimizer=Adam(), loss="mse")
    return model


def regression_accuracy(y_true, y_pred, tolerance=2):
    return np.mean(np.abs(y_true - y_pred) <= tolerance) * 100

@st.cache_resource
def run_regression_pipelines(merged_data):
    """Trains regression models, evaluates them, and returns the best pipelines along with original evaluation tables."""
    common_cols = [
        "school","sex","age","address","famsize","Pstatus",
        "Medu","Fedu","Mjob","Fjob","reason","nursery","internet"
    ]

    math_cols = [col for col in merged_data.columns if col.endswith("_mat")]
    math_cols.remove("G3_mat")

    por_cols = [col for col in merged_data.columns if col.endswith("_por")]
    por_cols.remove("G3_por")

    X_mat = merged_data[common_cols + math_cols]
    X_por = merged_data[common_cols + por_cols]

    y_mat = merged_data["G3_mat"]
    y_por = merged_data["G3_por"]

    cat_cols_mat = X_mat.select_dtypes(include="object").columns
    num_cols_mat = X_mat.select_dtypes(exclude="object").columns

    preprocessor_mat = ColumnTransformer([
        ("num", StandardScaler(), num_cols_mat),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_mat)
    ])

    cat_cols_por = X_por.select_dtypes(include="object").columns
    num_cols_por = X_por.select_dtypes(exclude="object").columns

    preprocessor_por = ColumnTransformer([
        ("num", StandardScaler(), num_cols_por),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_por)
    ])

    X_mat_train, X_mat_test, y_mat_train, y_mat_test = train_test_split(
        X_mat, y_mat, test_size=0.2, random_state=42
    )

    X_por_train, X_por_test, y_por_train, y_por_test = train_test_split(
        X_por, y_por, test_size=0.2, random_state=42
    )
    
    # Fit preprocessors for XAI transformations
    preprocessor_mat.fit(X_mat_train)
    preprocessor_por.fit(X_por_train)
    
    pipelines_mat = {
        "MLP": Pipeline([
            ("preprocess", preprocessor_mat),
            ("model", KerasRegressor(model=mlp_model, epochs=80, verbose=0))
        ]),
        "Autoencoder Regressor": Pipeline([
            ("preprocess", preprocessor_mat),
            ("model", KerasRegressor(model=autoencoder_reg, epochs=100, verbose=0))
        ]),
        "Denoising Autoencoder": Pipeline([
            ("preprocess", preprocessor_mat),
            ("model", KerasRegressor(model=denoising_autoencoder, epochs=100, verbose=0))
        ]),
        "Wide & Deep": Pipeline([
            ("preprocess", preprocessor_mat),
            ("model", KerasRegressor(model=wide_deep, epochs=80, verbose=0))
        ]),
        "Attention NN": Pipeline([
            ("preprocess", preprocessor_mat),
            ("model", KerasRegressor(model=attention_nn, epochs=80, verbose=0))
        ])
    }

    pipelines_por = {
        "MLP": Pipeline([
            ("preprocess", preprocessor_por),
            ("model", KerasRegressor(model=mlp_model, epochs=80, verbose=0))
        ]),
        "Autoencoder Regressor": Pipeline([
            ("preprocess", preprocessor_por),
            ("model", KerasRegressor(model=autoencoder_reg, epochs=100, verbose=0))
        ]),
        "Denoising Autoencoder": Pipeline([
            ("preprocess", preprocessor_por),
            ("model", KerasRegressor(model=denoising_autoencoder, epochs=100, verbose=0))
        ]),
        "Wide & Deep": Pipeline([
            ("preprocess", preprocessor_por),
            ("model", KerasRegressor(model=wide_deep, epochs=80, verbose=0))
        ]),
        "Attention NN": Pipeline([
            ("preprocess", preprocessor_por),
            ("model", KerasRegressor(model=attention_nn, epochs=80, verbose=0))
        ])
    }
    
    results = []

    for name, pipe in pipelines_mat.items():
        p = clone(pipe)
        p.fit(X_mat_train, y_mat_train)
        preds = p.predict(X_mat_test)
        acc = regression_accuracy(y_mat_test.values, preds)
        results.append([name, "Math", np.sqrt(mean_squared_error(y_mat_test, preds)), mean_absolute_error(y_mat_test, preds), r2_score(y_mat_test, preds), acc])

    for name, pipe in pipelines_por.items():
        p = clone(pipe)
        p.fit(X_por_train, y_por_train)
        preds = p.predict(X_por_test)
        acc = regression_accuracy(y_por_test.values, preds)
        results.append([name, "Portuguese", np.sqrt(mean_squared_error(y_por_test, preds)), mean_absolute_error(y_por_test, preds), r2_score(y_por_test, preds), acc])

    results_df = pd.DataFrame(results, columns=["Model", "Subject", "RMSE", "MAE", "R2", "Accuracy (±2)"])

    math_comparison = (
        results_df[results_df["Subject"] == "Math"]
        .sort_values(by=["Accuracy (±2)", "R2", "MAE", "RMSE"], ascending=[False, False, True, True])
        .reset_index(drop=True)
    )

    por_comparison = (
        results_df[results_df["Subject"] == "Portuguese"]
        .sort_values(by=["Accuracy (±2)", "R2", "MAE", "RMSE"], ascending=[False, False, True, True])
        .reset_index(drop=True)
    )

    best_math_model_name = math_comparison.iloc[0]["Model"]
    best_por_model_name = por_comparison.iloc[0]["Model"]

    final_math_pipeline = clone(pipelines_mat[best_math_model_name])
    final_por_pipeline = clone(pipelines_por[best_por_model_name])

    final_math_pipeline.fit(X_mat_train, y_mat_train)
    final_por_pipeline.fit(X_por_train, y_por_train)
    
    return {
        "final_math_pipeline": final_math_pipeline,
        "final_por_pipeline": final_por_pipeline,
        "math_comparison": math_comparison,
        "por_comparison": por_comparison,
        "results_df": results_df,
        "preprocessor_mat": preprocessor_mat,
        "preprocessor_por": preprocessor_por,
        "X_mat_train": X_mat_train,
        "X_por_train": X_por_train,
        "y_mat_train": y_mat_train,
        "y_por_train": y_por_train,
        "X_mat": X_mat, # for sample predict context
        "X_por": X_por, # for sample predict context
        "y_mat": y_mat,
        "y_por": y_por,
        "common_cols": common_cols,
        "math_cols": math_cols,
        "por_cols": por_cols
    }

from sklearn.base import BaseEstimator, RegressorMixin

class KerasWrapper(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.is_fitted_ = True
        
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def __sklearn_is_fitted__(self):
        return True



# ===============================
# CLASSIFICATION MODELS DEFINITION
# ===============================
def create_risk_label(row):
    avg_score = (row["G3_mat"] + row["G3_por"]) / 2
    if avg_score < 8:
        return "High Risk"
    elif avg_score < 12:
        return "Medium Risk"
    else:
        return "Low Risk"


def model_mlp_class(meta=None):
    n = meta["n_features_in_"]
    model = Sequential([
        Input(shape=(n,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(3, activation="softmax")
    ])
    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def model_cnn_class(meta=None):
    n = meta["n_features_in_"]
    model = Sequential([
        Input(shape=(n,)),
        Reshape((n,1)),
        Conv1D(32,3,activation="relu"),
        Flatten(),
        Dense(32,activation="relu"),
        Dense(3,activation="softmax")
    ])
    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def model_lstm_class(meta=None):
    n = meta["n_features_in_"]
    model = Sequential([
        Input(shape=(n,)),
        Reshape((n,1)),
        LSTM(32),
        Dense(3,activation="softmax")
    ])
    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def model_rnn_class(meta=None):
    n = meta["n_features_in_"]
    model = Sequential([
        Input(shape=(n,)),
        Reshape((n,1)),
        SimpleRNN(32),
        Dense(3,activation="softmax")
    ])
    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def model_autoencoder_class(meta=None):
    n = meta["n_features_in_"]
    inp = Input(shape=(n,))
    encoded = Dense(64, activation="relu")(inp)
    decoded = Dense(64, activation="relu")(encoded)
    out = Dense(3, activation="softmax")(decoded)
    model = Model(inp, out)
    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

@st.cache_resource
def run_classification_pipelines(merged_data):
    """Trains classification models, evaluates them, and returns the best pipeline."""
    merged_data_cls = merged_data.copy()
    merged_data_cls["Risk_Label"] = merged_data_cls.apply(create_risk_label, axis=1)

    X_class = merged_data_cls.drop(columns=["G3_mat", "G3_por", "Risk_Label"])
    
    # 3-class mapping: 0 = Low Risk, 1 = Medium Risk, 2 = High Risk
    risk_map = {"Low Risk": 0, "Medium Risk": 1, "High Risk": 2}
    y_class = merged_data_cls["Risk_Label"].map(risk_map).astype(int)

    cat_cols = X_class.select_dtypes(include="object").columns
    num_cols = X_class.select_dtypes(exclude="object").columns

    preprocessor_class = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_class, y_class,
        test_size=0.2,
        random_state=42,
        stratify=y_class
    )
    
    classifiers = {
        "DL MLP": KerasClassifier(model=model_mlp_class, epochs=60, batch_size=32, verbose=0),
        "DL CNN": KerasClassifier(model=model_cnn_class, epochs=60, batch_size=32, verbose=0),
        "DL LSTM": KerasClassifier(model=model_lstm_class, epochs=60, batch_size=32, verbose=0),
        "DL RNN": KerasClassifier(model=model_rnn_class, epochs=60, batch_size=32, verbose=0),
         "DL Autoencoder": KerasClassifier(model=model_autoencoder_class, epochs=60, batch_size=32, verbose=0)
    }

    classification_pipelines = {}
    classification_results = []

    for name, clf in classifiers.items():
        pipe = Pipeline([
            ("preprocess", preprocessor_class),
            ("model", clf)
        ])
        
        classification_pipelines[name] = pipe
        pipe.fit(X_train_cls, y_train_cls)
        y_pred = pipe.predict(X_test_cls)
        acc = accuracy_score(y_test_cls, y_pred)
        classification_results.append([name, acc])
        
    classification_results_df = pd.DataFrame(
        classification_results,
        columns=["Model", "Accuracy"]
    ).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    best_classifier_name = classification_results_df.iloc[0]["Model"]
    final_classifier = clone(classification_pipelines[best_classifier_name])
    final_classifier.fit(X_train_cls, y_train_cls)
    
    return {
        "final_classifier": final_classifier,
        "classification_results_df": classification_results_df,
        "X_class": X_class,
        "y_class": y_class,
        "X_train_cls": X_train_cls,
        "y_train_cls": y_train_cls,
        "preprocessor_class": preprocessor_class
    }


# ===============================
# PERSONALIZED RECOMMENDATION ENGINE
# ===============================

# Rich, subject-aware recommendation library.
# Keys: base feature name (no _mat/_por suffix)
# Values: dict with 'label', 'higher_is_better', 'math_advice', 'por_advice', 'common_advice'
# Each advice entry is a list of strings: [short_title, detail1, detail2, ...]
RECOMMENDATION_LIBRARY = {
    "absences": {
        "label": "Class Attendance",
        "higher_is_better": False,
        "math_actions": [
            "📅 Set a personal goal of zero unexcused absences for the next 4 weeks.",
            "📖 If you must miss class, review lesson notes the same evening using the textbook or a peer's notes.",
            "🔔 Use phone reminders or an alarm system to prevent oversleeping and tardiness.",
            "📊 Attendance directly predicts G3 — each missed Math class drops your predicted grade by ~0.3 points.",
        ],
        "por_actions": [
            "📅 Commit to attending every Portuguese lesson — oral participation is assessed each session.",
            "📝 Ask the teacher for a catch-up exercise if you do miss a class.",
            "🤝 Form a study pair who can share notes and alert you about assignments.",
            "📊 High absences in Portuguese strongly correlate with low G3 outcomes in this dataset.",
        ],
        "common_actions": [
            "🗓️ Track your own attendance weekly in a planner or spreadsheet.",
            "👪 Discuss attendance barriers with family or a school counselor.",
        ],
    },
    "failures": {
        "label": "Past Subject Failures",
        "higher_is_better": False,
        "math_actions": [
            "🔁 Identify the specific Math topics that caused previous failures (algebra, geometry, etc.) and target them first.",
            "🧮 Use Khan Academy or similar free resources for daily 20-minute focused practice on weak topics.",
            "🙋 Request extra help sessions from the Math teacher once per week.",
            "✅ Write down and re-do all past failed test questions as a structured revision exercise.",
        ],
        "por_actions": [
            "🔁 Review the grammar chapters you struggled with in previous years.",
            "📚 Read one short Portuguese text per day and summarize it in writing.",
            "🙋 Ask the Portuguese teacher for targeted feedback on your weakest areas.",
            "📝 Practice past exam writing prompts under timed conditions.",
        ],
        "common_actions": [
            "🧠 Understand that past failures are data — use them as a diagnostic, not a verdict.",
            "📋 Set a structured weekly revision plan addressing the root cause of each failure.",
        ],
    },
    "studytime": {
        "label": "Weekly Study Time",
        "higher_is_better": True,
        "math_actions": [
            "⏱️ Increase Math study time to at least 5 hours per week (from your current <2-5 hrs).",
            "📐 Spend at least 3 of those hours solving practice problems rather than just re-reading notes.",
            "📅 Schedule fixed Math study blocks (e.g., Monday, Wednesday, Friday for 1.5 hrs each).",
            "🔢 Focus problem-solving sessions on the topic type most likely to appear on the upcoming test.",
        ],
        "por_actions": [
            "⏱️ Dedicate at least 3–4 hours per week specifically to Portuguese practice.",
            "📖 Allocate 30 minutes daily to reading in Portuguese — newspaper, novel, or online articles.",
            "✍️ Spend one study session per week practising essay writing with timed conditions.",
        ],
        "common_actions": [
            "📵 Use a phone-free study zone to maximize effective study time.",
            "🍅 Try the Pomodoro technique: 25 min focused study + 5 min break for better retention.",
            "😴 Ensure 7–8 hours of sleep — sleep deprivation sharply reduces study effectiveness.",
        ],
    },
    "G2": {
        "label": "Period 2 Grade",
        "higher_is_better": True,
        "math_actions": [
            "📉 Your G2 Math grade is below average — the model heavily weights G2 when predicting G3.",
            "🔄 Thoroughly revise all G2 exam topics before the final exam.",
            "📋 Create a concept map of G2 topics, ranking them by difficulty, and address hardest first.",
            "🧮 Practice at least 5 exam-style problems per topic from your G2 syllabus.",
        ],
        "por_actions": [
            "📉 Your G2 Portuguese grade signals a risk for G3 — review all G2 essay and grammar feedback.",
            "📚 Re-read the literary texts covered in G2 and summarize their key themes.",
            "✍️ Write one practice essay per week on a G2 topic, then get teacher feedback.",
        ],
        "common_actions": [
            "📊 G2 is the strongest predictor of G3 in the dataset — improving G2 knowledge directly lifts your G3 chance.",
            "📆 Set a 2-week intensive revision plan focused entirely on G2 content.",
        ],
    },
    "G1": {
        "label": "Period 1 Grade",
        "higher_is_better": True,
        "math_actions": [
            "🔁 Low G1 Math grade — make sure to fill any foundational gaps before they compound.",
            "📘 Review your G1 Math exam topics, especially the ones marked incorrect.",
            "👩‍🏫 Ask the teacher for the G1 analysis report to identify specific weak concepts.",
        ],
        "por_actions": [
            "🔁 Low G1 Portuguese grade — foundational grammar and vocabulary gaps often persist to G3.",
            "📕 Re-study G1 grammar topics (conjugation, sentence structure) until fully comfortable.",
            "📖 Read a G1-level book/text and actively look up words you don't know.",
        ],
        "common_actions": [
            "💬 Discuss G1 performance with your teacher to get a targeted improvement plan.",
        ],
    },
    "goout": {
        "label": "Social Going-Out Frequency",
        "higher_is_better": False,
        "math_actions": [
            "⚖️ Reduce going-out nights to maximum 1–2 per week to free up time for Math practice.",
            "📱 Avoid social media and outings on evenings before tests or homework deadlines.",
        ],
        "por_actions": [
            "⚖️ Reserve weekend evenings for Portuguese reading or writing practice instead of going out.",
        ],
        "common_actions": [
            "🎯 Set a clear 'study first, social later' rule for weekdays.",
            "📊 Data shows students with high go-out frequency score on average 1.5 points lower in G3.",
            "🤝 Consider group study with friends as a social substitute that also builds academic skill.",
        ],
    },
    "Walc": {
        "label": "Weekend Alcohol Consumption",
        "higher_is_better": False,
        "math_actions": [
            "🚫 High weekend alcohol use negatively correlates with Math performance — reduce or eliminate.",
            "💧 Replace alcohol with physical activity on weekends to improve energy and cognitive focus.",
        ],
        "por_actions": [
            "🚫 Weekend alcohol consumption disrupts language retention — reduce for better Portuguese performance.",
        ],
        "common_actions": [
            "🧠 Alcohol impairs memory consolidation during sleep, directly hurting academic retention.",
            "💬 Speak with a school counselor if reducing alcohol is difficult.",
        ],
    },
    "Dalc": {
        "label": "Weekday Alcohol Consumption",
        "higher_is_better": False,
        "math_actions": [
            "🚫 Weekday alcohol use is a critical risk factor — it directly impairs problem-solving ability needed for Math.",
        ],
        "por_actions": [
            "🚫 Weekday alcohol reduces reading comprehension and writing focus — avoid during school days.",
        ],
        "common_actions": [
            "🏃 Replace weekday alcohol with a 20-min exercise routine to reduce stress naturally.",
            "💬 Seek support from a trusted adult or counselor if this is a recurring issue.",
        ],
    },
    "freetime": {
        "label": "After-School Free Time",
        "higher_is_better": False,
        "math_actions": [
            "⏳ Convert 1–2 hours of daily free time into structured Math revision.",
            "🎮 Use free time productively: math puzzle apps (e.g., Brilliant.org) instead of passive entertainment.",
        ],
        "por_actions": [
            "⏳ Dedicate 30 minutes of free-time to reading Portuguese texts or watching Portuguese content.",
        ],
        "common_actions": [
            "🗓️ Create a structured daily schedule — define both study and leisure blocks.",
            "⚽ Physical activity in free time is beneficial, but limit unstructured screen time.",
        ],
    },
    "traveltime": {
        "label": "School Travel Time",
        "higher_is_better": False,
        "math_actions": [
            "🗂️ Use commute time productively — review Math formula cards or watch explanation videos.",
        ],
        "por_actions": [
            "🎧 Listen to Portuguese audio content (podcasts, audiobooks) during your commute.",
        ],
        "common_actions": [
            "🚌 Convert travel time to micro-study sessions using flashcard apps like Anki.",
            "🏘️ If possible, explore carpooling/shorter routes to save study time at home.",
        ],
    },
    "health": {
        "label": "Student Health Status",
        "higher_is_better": True,
        "math_actions": [
            "🩺 Poor health reduces Math performance — prioritize sleep, nutrition, and exercise.",
            "😴 Aim for 8 hours of sleep before exams — memory consolidation happens during sleep.",
        ],
        "por_actions": [
            "🩺 Health directly affects cognitive ability — ensure regular meals and hydration.",
        ],
        "common_actions": [
            "🏃 Exercise at least 3 times per week — it significantly boosts cognitive performance.",
            "🍎 Maintain a balanced diet with brain-boosting foods (nuts, fish, leafy greens).",
            "💬 Reach out to the school nurse or counselor if health issues are affecting your studies.",
        ],
    },
    "schoolsup": {
        "label": "School Extra Support",
        "higher_is_better": True,
        "math_actions": [
            "🏫 If not already enrolled, ask for extra Math support classes at school.",
            "📝 Make use of after-school tutoring sessions offered by the math department.",
        ],
        "por_actions": [
            "🏫 Request additional Portuguese language support sessions at school.",
            "📖 Use school library resources for Portuguese language practice.",
        ],
        "common_actions": [
            "🙋 Don't hesitate to ask teachers for additional materials or study guidance.",
            "💻 Explore online school support platforms your institution may provide.",
        ],
    },
    "famsup": {
        "label": "Family Academic Support",
        "higher_is_better": True,
        "math_actions": [
            "👨‍👩‍👦 Ask a family member to quiz you on Math formulas or review your homework with you.",
        ],
        "por_actions": [
            "👨‍👩‍👦 Ask a family member to read Portuguese texts aloud with you or discuss topics together.",
        ],
        "common_actions": [
            "💬 Share your academic goals with your family — accountability improves performance.",
            "📱 Use family group chats to share study milestones and get encouragement.",
        ],
    },
    "Medu": {
        "label": "Mother's Education Level",
        "higher_is_better": True,
        "math_actions": [],
        "por_actions": [],
        "common_actions": [
            "🎓 If parental educational background is limited, seek mentorship from a teacher or tutor who can fill that gap.",
            "📚 Look for free online resources (Khan Academy, YouTube) as supplements for home-based learning support.",
        ],
    },
    "Fedu": {
        "label": "Father's Education Level",
        "higher_is_better": True,
        "math_actions": [],
        "por_actions": [],
        "common_actions": [
            "🎓 Seek mentorship from school staff or community programs to supplement parental academic guidance.",
            "📚 Explore free tutoring through community centers or online volunteer tutoring services.",
        ],
    },
}

# Priority thresholds for styling
SHAP_THRESHOLD_CRITICAL  = 0.5   # |SHAP| > this → Critical
SHAP_THRESHOLD_IMPORTANT = 0.2   # |SHAP| > this → Important


def _clean_base(feature_name: str) -> str:
    """Strip _mat / _por suffix to get base feature name."""
    if feature_name.endswith("_mat") or feature_name.endswith("_por"):
        return feature_name.replace("_mat", "").replace("_por", "")
    return feature_name


def _priority_label(shap_magnitude: float) -> str:
    if shap_magnitude >= SHAP_THRESHOLD_CRITICAL:
        return "critical"
    elif shap_magnitude >= SHAP_THRESHOLD_IMPORTANT:
        return "important"
    return "helpful"


def generate_rich_recommendations(
    student_row: pd.Series,
    avg_features: pd.Series,
    shap_mat_df=None,   # pd.DataFrame with cols ["Feature", "SHAP Value"] — negative = hurts grade
    shap_por_df=None,   # same for Portuguese
) -> dict:
    risk_level  = str(student_row.get("Pred_Risk_Level", "Unknown"))
    math_grade  = float(student_row.get("Pred_G3_Math", 0))
    por_grade   = float(student_row.get("Pred_G3_Portuguese", 0))

    math_actions   = []
    por_actions    = []
    common_actions_dict = {}  # feature -> entry (deduplicated)

    # ── Helper: extract top-N negative SHAP contributors ──────────────────────
    def top_negative_shap(shap_df, n=5) -> list[dict]:
        """Return top-N features where SHAP < 0 (hurting the grade), sorted by abs magnitude."""
        if shap_df is None or shap_df.empty:
            return []
        neg = shap_df[shap_df["SHAP Value"] < 0].copy()
        neg["AbsMag"] = neg["SHAP Value"].abs()
        return neg.sort_values("AbsMag", ascending=False).head(n).to_dict("records")

    # ── Math recommendations ───────────────────────────────────────────────────
    for entry in top_negative_shap(shap_mat_df):
        raw_feat  = entry["Feature"]
        base_feat = _clean_base(raw_feat)
        shap_mag  = abs(entry["SHAP Value"])
        if base_feat in RECOMMENDATION_LIBRARY:
            lib  = RECOMMENDATION_LIBRARY[base_feat]
            steps = lib["math_actions"] or lib["common_actions"]
            if steps:
                math_actions.append({
                    "feature": base_feat,
                    "label":   lib["label"],
                    "priority": _priority_label(shap_mag),
                    "steps":   steps,
                    "shap":    shap_mag,
                })
                # add common actions to shared pool
                if lib["common_actions"] and base_feat not in common_actions_dict:
                    common_actions_dict[base_feat] = {
                        "feature": base_feat, "label": lib["label"],
                        "steps": lib["common_actions"]
                    }

    # ── Portuguese recommendations ─────────────────────────────────────────────
    for entry in top_negative_shap(shap_por_df):
        raw_feat  = entry["Feature"]
        base_feat = _clean_base(raw_feat)
        shap_mag  = abs(entry["SHAP Value"])
        if base_feat in RECOMMENDATION_LIBRARY:
            lib  = RECOMMENDATION_LIBRARY[base_feat]
            steps = lib["por_actions"] or lib["common_actions"]
            if steps:
                por_actions.append({
                    "feature": base_feat,
                    "label":   lib["label"],
                    "priority": _priority_label(shap_mag),
                    "steps":   steps,
                    "shap":    shap_mag,
                })
                if lib["common_actions"] and base_feat not in common_actions_dict:
                    common_actions_dict[base_feat] = {
                        "feature": base_feat, "label": lib["label"],
                        "steps": lib["common_actions"]
                    }

    # ── Fallback: use avg-feature comparison when SHAP not available ───────────
    if not math_actions and not por_actions:
        lower_better = {"absences_mat","absences_por","failures_mat","failures_por",
                        "goout","Walc","Dalc","freetime","traveltime"}
        diffs: list[tuple[str, str, float]] = []
        for feat in student_row.index:
            if feat in avg_features.index and pd.api.types.is_numeric_dtype(type(student_row[feat])):
                base = _clean_base(feat)
                s_val = float(student_row[feat])
                a_val = float(avg_features[feat])
                if feat in lower_better:
                    diff = s_val - a_val
                else:
                    diff = a_val - s_val
                if diff > 0 and base in RECOMMENDATION_LIBRARY:
                    diffs.append((feat, base, diff))
        diffs.sort(key=lambda x: float(x[2]), reverse=True)
        for raw_feat, base_feat, diff in [diffs[i] for i in range(min(4, len(diffs)))]:
            lib = RECOMMENDATION_LIBRARY[base_feat]
            if raw_feat.endswith("_mat") or not raw_feat.endswith("_por"):
                steps = lib["math_actions"] or lib["common_actions"]
                if steps:
                    math_actions.append({"feature": base_feat, "label": lib["label"],
                                         "priority": "important", "steps": steps, "shap": diff})
            if raw_feat.endswith("_por") or not raw_feat.endswith("_mat"):
                steps = lib["por_actions"] or lib["common_actions"]
                if steps:
                    por_actions.append({"feature": base_feat, "label": lib["label"],
                                        "priority": "important", "steps": steps, "shap": diff})
            if base_feat not in common_actions_dict and lib["common_actions"]:
                common_actions_dict[base_feat] = {
                    "feature": base_feat, "label": lib["label"],
                    "steps": lib["common_actions"]
                }

    # ── Estimate improvement potential ─────────────────────────────────────────
    math_potential = float(f"{max(0.0, 20.0 - math_grade) * 0.3:.2f}")   # conservative ~30% headroom
    por_potential  = float(f"{max(0.0, 20.0 - por_grade) * 0.3:.2f}")

    # ── Summary text ───────────────────────────────────────────────────────────
    weaker_subject = "Mathematics" if math_grade < por_grade else "Portuguese"
    if risk_level == "High Risk":
        summary_text = (
            f"⚠️ This student is at **High Risk** with Math={math_grade:.1f} and Portuguese={por_grade:.1f}. "
            f"Immediate, targeted intervention is required. Focus first on **{weaker_subject}** while addressing "
            f"the top-{len(math_actions)} Math and top-{len(por_actions)} Portuguese issues identified below."
        )
    elif risk_level == "Medium Risk":
        summary_text = (
            f"🟡 This student is at **Medium Risk** (Math={math_grade:.1f}, Portuguese={por_grade:.1f}). "
            f"With consistent effort on the recommended actions, grade improvement of up to "
            f"+{math_potential} (Math) and +{por_potential} (Portuguese) is possible."
        )
    else:
        summary_text = (
            f"✅ This student is **Low Risk** (Math={math_grade:.1f}, Portuguese={por_grade:.1f}). "
            "Keep maintaining current study habits and aim for even higher performance!"
        )

    return {
        "risk_level":          risk_level,
        "math_grade":          math_grade,
        "por_grade":           por_grade,
        "math_actions":        math_actions,
        "por_actions":         por_actions,
        "common_actions":      list(common_actions_dict.values()),
        "improvement_potential": {"math": math_potential, "por": por_potential},
        "summary_text":        summary_text,
    }


# ── Backward-compatible wrapper (used by 3_Dataset_Predictions.py) ─────────────
def clean_feature_name(feature):
    """Strip _mat / _por suffix."""
    return _clean_base(feature)


def generate_recommendations(row, avg_features):
    """
    Lightweight wrapper that returns a pipe-separated string of advice
    (kept for backward compatibility with Dataset Predictions page).
    """
    rec_dict  = generate_rich_recommendations(row, avg_features)
    risk      = rec_dict["risk_level"]
    math_g    = rec_dict["math_grade"]
    por_g     = rec_dict["por_grade"]

    parts = []
    if risk == "High Risk":
        for act in (rec_dict["math_actions"] + rec_dict["por_actions"])[:3]:
            if act["steps"]:
                parts.append(act["steps"][0])   # first step of each top action
        if math_g < por_g:
            parts.append("Focus more effort on improving Mathematics performance")
        else:
            parts.append("Focus more effort on improving Portuguese performance")
        parts.append("Immediate improvement plan is required to avoid failing")
    elif risk == "Medium Risk":
        parts.append("Consider reviewing recent topics to improve overall grades")
        if math_g < 10:
            parts.append("Dedicate more time to Mathematics")
        if por_g < 10:
            parts.append("Dedicate more time to Portuguese")
    else:
        parts.append("Maintain current study habits — performance is on track")

    return " | ".join(parts) if parts else "No specific recommendations at this time."
