def load_css():

    return """
<style>

/********************
BACKGROUND
********************/

.stApp{
    background:#0f172a;
}

/********************
SIDEBAR
********************/

section[data-testid="stSidebar"]{
    background:#111827;
    border-right:1px solid #262626;
}

/********************
CHAT INPUT
********************/

.stChatInputContainer{

    border:none;

    background:#0f172a;

}

.stChatInput{

    border-radius:20px;

}

/********************
BUTTON
********************/

.stButton>button{

    width:100%;

    border-radius:12px;

    height:45px;

    border:1px solid #333;

    background:#1f2937;

}

.stButton>button:hover{

    background:#374151;

}

/********************
MESSAGE
********************/

[data-testid="stChatMessage"]{

    border-radius:18px;

    padding:10px;

}

/********************
HEADINGS
********************/

h1{

    font-weight:800;

}

h2{

    font-weight:700;

}

/********************
REMOVE STREAMLIT
********************/

#MainMenu{

visibility:hidden;

}

footer{

visibility:hidden;

}

header{

visibility:hidden;

}

</style>
"""