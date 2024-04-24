import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io

# Fungsi untuk membuat model regresi linier
def create_linear_regression_model(X, Y):
    model = LinearRegression()
    model.fit(np.array(X).reshape(-1, 1), Y)
    return model

# Fungsi untuk menampilkan persamaan regresi linier dan nilai koefisien korelasi
def display_regression_equation(X, Y, model):
    a = model.intercept_
    b = model.coef_[0]
    r = model.score(np.array(X).reshape(-1, 1), Y)  # Koefisien korelasi
    equation = f'y = {a:.2f} + {b:.2f}x'
    regression_info = {'equation': equation, 'intercept': a, 'slope': b, 'r_value': r}
    return regression_info

# Fungsi untuk menampilkan grafik linearitas
def display_linear_plot(X, Y, model, label_X, label_Y):
    fig, ax = plt.subplots()
    ax.scatter(X, Y, color='blue', label='Data Asli')
    ax.plot(X, model.predict(np.array(X).reshape(-1, 1)), color='red', label='Regresi Linier')
    ax.set_xlabel(label_X)
    ax.set_ylabel(label_Y)
    ax.legend()
    st.pyplot(fig)

# Halaman aplikasi Streamlit
def main():
    st.title('Penentuan Grafik dan Persamaan Regresi Linearlitas')

    st.write('Masukkan data X dan Y untuk membuat model regresi linier')

    # Input data X dan Y dari pengguna
    X_input = st.text_input('Masukkan nilai X (pisahkan dengan koma jika lebih dari satu):').strip()
    Y_input = st.text_input('Masukkan nilai Y (pisahkan dengan koma jika lebih dari satu):').strip()

    label_X = st.text_input('Masukkan label untuk sumbu X:')
    label_Y = st.text_input('Masukkan label untuk sumbu Y:')

    if X_input and Y_input and label_X and label_Y:
        X = [float(x) for x in X_input.split(',')]
        Y = [float(y) for y in Y_input.split(',')]

        # Membuat model regresi linier
        model = create_linear_regression_model(X, Y)

        # Menampilkan grafik linearitas
        display_linear_plot(X, Y, model, label_X, label_Y)

        # Menampilkan persamaan regresi linier, nilai slope (b), nilai intercept (a), dan nilai koefisien korelasi (r)
        regression_info = display_regression_equation(X, Y, model)
        st.markdown('### Persamaan Regresi Linier:')
        st.markdown(f'```{regression_info["equation"]}```', unsafe_allow_html=True)

        st.markdown('### Nilai Slope (b):')
        st.write(f'{regression_info["slope"]:.2f}')

        st.markdown('### Nilai Intercept (a):')
        st.write(f'{regression_info["intercept"]:.2f}')

        st.markdown('### Nilai Koefisien Korelasi (r):')
        st.write(f'{regression_info["r_value"]:.4f}')

if __name__ == '__main__':
    main()
