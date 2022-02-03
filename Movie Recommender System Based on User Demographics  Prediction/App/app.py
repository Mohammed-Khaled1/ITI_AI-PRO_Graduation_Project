###################### Importing Libraries ######################

from flask import Flask, g, redirect, render_template, request, session, url_for

from flask_mysqldb import MySQL
# from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
import emoji
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import cv2
import time
import sys
import os

from keras.models import load_model

###########################################################################

# To make camera window active
os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''') 

###########################################################################

# User class for the users in the app
# class User:
#     def __init__(self, id, username, password):
#         self.id = id
#         self.username = username
#         self.password = password

#     def __repr__(self):
#         return f'<User: {self.username}>'

# users = []
# users.append(User(id=1, username='Kareem', password='123'))
# users.append(User(id=2, username='Alex', password='abc'))
# users.append(User(3, 'Mike', 'xyz'))

# print(users)

###################### APP Creation & Configurations ######################

# Create an instance of Flask class for the web app
app = Flask(__name__)
app.secret_key = 'secretKey1'
mail = Mail(app) # instantiate the mail class

# Configuring the web sql serverd
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'graduation_project'
mysql = MySQL(app)

# configuration of mail
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'recommenderteam6@gmail.com'
app.config['MAIL_PASSWORD'] = 'RecoTeam_666'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

###################### CONSTANT & GLOBAL VARIABLES ######################

# Keras models path
AGE_MODEL_PATH = "models/aaage_best_model_last6.h5"
ETHNICITY_MODEL_PATH = "models/Adagrad_test_ethnicity_model.h5"
GENDER_MODEL_PATH = "models/gender_best_model.h5"

# Photos Path
FACE_DETECTOR_PATH = 'static/haarcascade_frontalface_default.xml'
USER_PHOTO_PATH = "static/user_photos/"
GUEST_PHOTO_PATH = "static/guest_photos/"

#### Model Labels ####

# Age labels
AGE_LABELS = ['Under 25', '25-34', '35-44', '45-55', '55+']
DF_AGE_LABELS = ['18', '25', '35', '45', '50', '56']

# Ethnicity labels
ETHNICITY_LABELS = ['American-European', 'African', 'Japanese', 'Indian', 'Latin']

# Gender labels
GENDER_LABELS = ['Male', 'Female']

# Movie Genres
MOVIE_GENRES = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War']

# OTP CODE
OTP = ''

#########################################################################

# Get all App users
# def get_all_users():
#     cur = mysql.connection.cursor()
#     cur.execute("SElECT * FROM users_info")
#     all_users = cur.fetchall()
#     cur.close()

#     return all_users

#######################################################

# g object is a global variable for one specific request.
# It is used to store data b\w different functions without passing them directly.
@app.before_request
def before_request():
    g.user = None

    if 'user_id' in session:
        # Get all users from database
        # cur = mysql.connection.cursor()
        # cur.execute("SElECT * FROM users_info")
        # all_users = cur.fetchall()
        # cur.close()

        # user = [user for user in users if user.id == session['user_id']][0]
        # # print('inside before request :', user)
        # g.user = user

        # all_users = get_all_users()
        # user = [user for user in all_users if user[0] == session['user_id']][0]
        # g.user = user

        ##################################### Other solution
        # print('################# inside before_request', file=sys.stderr)
        # print('type(session["user_id"]) =', type(session), file=sys.stderr)
        # print('session["user_id"] =', session['user_id'], file=sys.stderr)
        # print('session["user_id"][0]', session['user_id'][0], file=sys.stderr)

        # Get the user record from the database
        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM users_info WHERE id = %s", (str(session['user_id']),))
            user = cur.fetchall()[0]
            cur.close()

            # print('after query in before_request', file=sys.stderr)
            # print('type(user) =', type(user), file=sys.stderr)
            # print('user =', user, file=sys.stderr)
            # print('user[0] =', user[0], file=sys.stderr)

            g.user = user

        except:
            print('No users with this id', file=sys.stderr)

            try:
                cur = mysql.connection.cursor()
                cur.execute("SELECT * FROM users_info WHERE Email = %s", (str(session['user_id']),))
                user = cur.fetchall()[0]
                cur.close()

                # print('after query in before_request', file=sys.stderr)
                # print('type(user) =', type(user), file=sys.stderr)
                # print('user =', user, file=sys.stderr)
                # print('user[0] =', user[0], file=sys.stderr)

                g.user = user

            except:
                print('No users with this Email (Forget Password)', file=sys.stderr)

#######################################################

# the root page redirects to the login page 
@app.route('/')
def mainPage():
    return redirect(url_for('login_signup'))

# Login + Signup Page
@app.route('/login_signup', methods=['GET', 'POST'])
def login_signup(): 
    msg = ""
    confirmed = 1

    if request.method == 'POST':
        # Clear session
        session.pop('user_id', None)

        ##### Log in Page #####
        if "btn_login_submit" in request.form:
            # Login variables
            logEmail = request.form['logEmail']
            logPass = request.form['logPass']

            # if logEmail != '' and logPass != '':
                # Get all users from database
                # cur = mysql.connection.cursor()
                # cur.execute("SElECT * FROM users_info")
                # all_users = cur.fetchall()
                # cur.close()

                #####################################
                # all_users = get_all_users()
                # user = [user for user in all_users if user[3] == logEmail][0]
                # print('inside login function', user)
                # if user and user[4] == logPass:
                #     session['user_id'] = user[0]
                #     return redirect(url_for('profile'))

                # return redirect(url_for('login_signup'))
                #####################################
        
            ##################################### Other solution
            try:
                # Get the user record from the database
                cur = mysql.connection.cursor()
                cur.execute("SELECT * FROM users_info WHERE Email = %s", (logEmail,))
                curr_user = cur.fetchall()[0]
                cur.close()
                
                # print('################### inside login', file=sys.stderr)
                # print('type(curr_user) =', type(curr_user), file=sys.stderr)
                # print('curr_user =', curr_user, file=sys.stderr)
                # print('curr_user[0] =', curr_user[0], file=sys.stderr)

                # Check user password
                if curr_user[4] == logPass:
                    session['user_id'] = curr_user[0]

                    # print('################### inside try if password', file=sys.stderr)
                    # print('type(session["user_id"]) =', type(session['user_id']), file=sys.stderr)
                    # print('session["user_id"]', session['user_id'], file=sys.stderr)
                    # print('session["user_id"][0]', session['user_id'][0], file=sys.stderr)
                    
                    return redirect(url_for('home'))
                
            except:
                msg = 'Incorrect Email or Password !!'

                # print('################### inside except', file=sys.stderr)
                # print(curr_user, file=sys.stderr)

                return render_template('login_signup.html', message=msg)
            ####################################################### end of Other solution
            
        ##### Sign up Page #####
        if "btn_reg_submit" in request.form:
            # Signup variables
            details = request.form
            regFirstName = details['regFirstName']
            regLastName = details['regLastName']
            regEmail = details['regEmail']
            regPass = details['regPass'] 
            regConfPass = details['regConfPass']
            # genre = details['genre']
            genre = 'Comedy' # Temporary

            # print('################### inside login', file=sys.stderr)
            # print('logEmail =', logEmail, file=sys.stderr)
            # print('logPass =', logPass, file=sys.stderr)

            ##############################
            # if regFirstName != '' and regLastName != '' and regEmail != '' and regPass != '' and regConfPass != '' and genre != '':
                # if password != conf_password:
                #     msg = "Password mismatch"
                #     return render_template('signup.html', dataToRender=msg)
            ##############################

            # Take a photo after clicking register button
            photo_path = USER_PHOTO_PATH + regFirstName + '_' + str(int(time.time())) + '_photo.jpg'
            # take_photo_timer(photo_path)
            # take_photo_detect_face(photo_path)
            take_photo_timer_detect_face(photo_path)

            # Predict age, gender & age for the user from the photo taken
            age = model_pred(AGE_MODEL_PATH, AGE_LABELS, photo_path)
            ethnicity = model_pred(ETHNICITY_MODEL_PATH, ETHNICITY_LABELS, photo_path)
            gender = model_pred(GENDER_MODEL_PATH, GENDER_LABELS, photo_path)

            ######################################################
            # Send OTP code
            # if request.method == 'POST':
            #     if request.form['send_code'] == 'Do Something':
            #          # Create OTP
            #         otp = str(random.randint(0, 999)).zfill(6)

            #         # Send verification code
            #         verify_mail(email, otp)

            # # Check user verification code
            # otp_user_entry = details['verify']
            # if otp != conf_password:
            #     confirmed = 0
            #     msg = "Invalid Verification Code"
            #     return render_template('signup.html', dataToRender=msg)
            ######################################################
            
            try:
                # Insert the data to the database
                cur = mysql.connection.cursor()
                cur.execute("INSERT INTO users(First_Name, Last_Name, Email, Password) VALUES (%s, %s, %s, %s)", (regFirstName, regLastName, regEmail, regPass))
                cur.execute("INSERT INTO users_info(First_Name, Last_Name, Email, Password, Genre, Age, Ethnicity, Gender, Photo_path, Confirmed) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (regFirstName, regLastName, regEmail, regPass, genre, age, ethnicity, gender, photo_path, confirmed))
                mysql.connection.commit()
                cur.close()

                return redirect(url_for('login_signup'))

            except:
                # print('inside except', file=sys.stderr)
                # print(curr_user, file=sys.stderr)
                return render_template('login_signup.html')


    return render_template('login_signup.html')
        
##########################################################################

# Forget password page
@app.route('/forget',methods=['GET', 'POST'])
def forget():
    confirmed = 0
    msg = ''
    global OTP
    
    if request.method == 'POST':

        email = request.form['email']
        # print('################### Inside POST ', file=sys.stderr)
        # print('email =', email, file=sys.stderr)

        try:
            # check if the email is in the database
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM users WHERE Email = %s", (email,) )
            cur.execute("SELECT * FROM users_info WHERE Email = %s", (email,) )
            curr_user = cur.fetchall()[0]
            cur.close()

            if curr_user == ():
                msg = 'Email not found'
                return render_template('forget.html', message=msg) 

        except:
            msg = 'Email not found'
            # print('################### Inside Except DB ERROR', file=sys.stderr)

            return render_template('forget.html', message=msg) 

        if "btn_verify_submit" in request.form:
            # Check user verification code
            otp_user_entry = request.form['verify_otp']

            if OTP != otp_user_entry:
                confirmed = 0
                msg = "Invalid Verification Code"
                # print('################### OTP not matched ', file=sys.stderr)

                return render_template('forget.html', message=msg)

            else:
                confirmed = 1
                session['user_id'] = email
                # print('################### OTP matched ', file=sys.stderr)
                # print('session["user_id"] =', session['user_id'], file=sys.stderr)
                # print('g.user[0] =', g.user[0], file=sys.stderr)

                return redirect(url_for('change_pass'))

        if "btn_resend_submit" in request.form:
            # Create OTP
            OTP = str(random.randint(0, 999)).zfill(6)

            # Send verification code
            verify_mail(email, OTP)
            # print('################### OTP Resend', file=sys.stderr)

            return render_template('forget.html', storedEmail=email)

    ######################################################
    
    return render_template('forget.html')

#######################################################

# Change Password Page
@app.route('/change_pass', methods=['GET', 'POST'])
def change_pass():
    if not g.user:
        return redirect(url_for('login_signup'))

    if request.method == "POST" : 
        
        email = g.user[3]
        password = request.form['regConfPass']

        # Update the record in the database
        cur = mysql.connection.cursor()
        cur.execute("UPDATE users SET Password = %s WHERE Email = %s", (password, email) )
        cur.execute("UPDATE users_info SET Password = %s WHERE Email = %s", (password, email) )
        mysql.connection.commit()
        cur.close()

        return redirect(url_for('login_signup'))

    return render_template('change_pass.html')

#######################################################

# Home + Recommendation Page 
@app.route('/home', methods=['GET', 'POST'])
def home():
    if not g.user:
        return redirect(url_for('login_signup'))

    if request.method == "POST" :
        movie_genres = request.form['movie_genres']
        # print('################## movie_genres\n', movie_genres, file=sys.stderr)

        #################################################################
        # get all movies and users interactions from the database
        cur = mysql.connection.cursor()

        # cur.execute("SELECT Genre, Age, Ethnicity, Gender FROM users_info where id = %s", (g.user[0],))
        # cur.execute("SElECT Genre, Age, Ethnicity, Gender FROM users_info where id = %s", (7,))
        cur.execute("SELECT * FROM users_interactions")
        all_users_interactions = cur.fetchall()
        
        cur.execute("SELECT * FROM movies")
        all_movies = cur.fetchall()
        
        cur.close()

        # Convert the output to the dataframe
        users_interactions_df = pd.DataFrame(all_users_interactions)
        users_interactions_df.columns = ['bucketized_user_age', 'movie_genres', 'movie_id', 'movie_title', 'user_gender', 'user_id', 'user_rating', 'year']
        
        all_movies_df = pd.DataFrame(all_movies)
        all_movies_df.columns = ['movie_id', 'movie_genres', 'movie_title']
        # print('################## users_interactions_df\n', users_interactions_df, file=sys.stderr)
        # print('################## all_movies_df\n', all_movies_df, file=sys.stderr)

        # df_size
        df_size = pd.DataFrame(users_interactions_df.groupby('movie_id').size().sort_values(ascending=False))
        list(df_size.columns)

        df_size['size'] = df_size[0]
        df_size.drop(columns=[0],inplace=True)
        df_size = df_size[df_size['size']>50]
        most_pop=list(df_size.index)
        # print('################## df_size\n', df_size, file=sys.stderr)

        # Convert the age range into a single age within the range
        user_age_range = g.user[6]
        index = AGE_LABELS.index(user_age_range)
        age = DF_AGE_LABELS[index]
        # print('################## age\n', age, file=sys.stderr)

        # Get recommendations for the user
        recommendations = recommend_movies(n_top=10, age=int(age), gender=g.user[8], genre=movie_genres, df_ratings_final=users_interactions_df, df_size=df_size, df_movies=all_movies_df)
        # recommendations = recommend_movies(n_top=20, age=18, gender='Male', genre=['Action'], df_ratings_final=users_interactions_df, df_size=df_size, df_movies=all_movies_df)
        # print('################## recommendations\n', recommendations, file=sys.stderr)

        # Send recommendations to the user's email
        send_recommendations_to_mail(g.user, movie_genres, recommendations)

        return render_template('recommended_movies.html', len=len(recommendations), recommendations=recommendations)

        #######################

    return render_template('home.html', len=len(MOVIE_GENRES), MOVIE_GENRES=MOVIE_GENRES)

#######################################################

# Profile Page
@app.route('/profile')
def profile():
    if not g.user:
        return redirect(url_for('login_signup'))

    return render_template('profile.html')

#######################################################

# Edit profile page
@app.route('/profile_edit',  methods=['GET', 'POST'])
def profile_edit():
    msg = ''

    if not g.user:
        return redirect(url_for('login_signup'))

    if request.method == "POST" : 
        user_id = g.user[0]
        details = request.form
        first_name = details['fname']
        last_name = details['lname']
        email = details['email']
        password = g.user[4]
        age = details['age_labels_opt'] 
        ethnicity = details['ethnicity_labels_opt']
        gender = details['gender_labels_opt']
        photo_path = g.user[9]

        # Update the record in the database
        cur = mysql.connection.cursor()
        cur.execute("UPDATE users SET First_Name = %s, Last_Name = %s, Email = %s, Password = %s WHERE id = %s", (first_name, last_name, email, password, user_id))
        cur.execute("UPDATE users_info SET First_Name = %s, Last_Name = %s, Email = %s, Password = %s, Age = %s, Ethnicity = %s, Gender = %s, Photo_path = %s WHERE id = %s", (first_name, last_name, email, password, age, ethnicity, gender, photo_path, user_id))
        mysql.connection.commit()
        cur.close()

        msg = 'Changes Saved Successfully'

        return render_template('profile_edit.html', len=len(GENDER_LABELS), len_ETHNICITY=len(ETHNICITY_LABELS), len_AGE_LABELS=len(AGE_LABELS), GENDER_LABELS=GENDER_LABELS, ETHNICITY_LABELS=ETHNICITY_LABELS, AGE_LABELS=AGE_LABELS, Message=msg)
    
    return render_template('profile_edit.html', len=len(GENDER_LABELS), len_ETHNICITY=len(ETHNICITY_LABELS), len_AGE_LABELS=len(AGE_LABELS), GENDER_LABELS=GENDER_LABELS, ETHNICITY_LABELS=ETHNICITY_LABELS, AGE_LABELS=AGE_LABELS, Message=msg)

#######################################################

# Home + Recommendation  for Guests Page 
@app.route('/home_guest', methods=['GET', 'POST'])
def home_guest():

    #################################################################
    # get all movies and users interactions from the database
    cur = mysql.connection.cursor()

    # cur.execute("SELECT Genre, Age, Ethnicity, Gender FROM users_info where id = %s", (g.user[0],))
    # cur.execute("SElECT Genre, Age, Ethnicity, Gender FROM users_info where id = %s", (7,))
    cur.execute("SELECT * FROM users_interactions")
    all_users_interactions = cur.fetchall()
    
    cur.execute("SELECT * FROM movies")
    all_movies = cur.fetchall()
    
    cur.close()

    # Convert the output to the dataframe
    users_interactions_df = pd.DataFrame(all_users_interactions)
    users_interactions_df.columns = ['bucketized_user_age', 'movie_genres', 'movie_id', 'movie_title', 'user_gender', 'user_id', 'user_rating', 'year']
    
    all_movies_df = pd.DataFrame(all_movies)
    all_movies_df.columns = ['movie_id', 'movie_genres', 'movie_title']
    # print('################## users_interactions_df\n', users_interactions_df, file=sys.stderr)
    # print('################## all_movies_df\n', all_movies_df, file=sys.stderr)
    
    # df_size
    df_size = pd.DataFrame(users_interactions_df.groupby('movie_id').size().sort_values(ascending=False))
    list(df_size.columns)

    df_size['size'] = df_size[0]
    df_size.drop(columns=[0],inplace=True)
    df_size = df_size[df_size['size']>50]
    most_pop=list(df_size.index)
    # print('################## df_size\n', df_size, file=sys.stderr)

    if request.method == "POST" :
        movie_genres = request.form['movie_genres']
        # print('################## movie_genres\n', movie_genres, file=sys.stderr)

        # Take a photo after clicking register button
        photo_path = GUEST_PHOTO_PATH + 'guest_' + str(int(time.time())) + '_photo.jpg'
        # take_photo_timer(photo_path)
        # take_photo_detect_face(photo_path)
        take_photo_timer_detect_face(photo_path)

        # Predict age, gender & age for the user from the photo taken
        age = model_pred(AGE_MODEL_PATH, AGE_LABELS, photo_path)
        ethnicity = model_pred(ETHNICITY_MODEL_PATH, ETHNICITY_LABELS, photo_path)
        gender = model_pred(GENDER_MODEL_PATH, GENDER_LABELS, photo_path)

        # Convert the age range into a single age within the range
        user_age_range = age
        index = AGE_LABELS.index(user_age_range)
        age = DF_AGE_LABELS[index]
        # print('################## age\n', age, file=sys.stderr)
        # print('################## gender\n', gender, file=sys.stderr)

        # Get recommendations for the guest
        recommendations = recommend_movies(n_top=5, age=int(age), gender=gender, genre=movie_genres, df_ratings_final=users_interactions_df, df_size=df_size, df_movies=all_movies_df)
        # recommendations = recommend_movies(n_top=5, age=18, gender='Male', genre=['Action'], df_ratings_final=users_interactions_df, df_size=df_size, df_movies=all_movies_df)
        # print('################## recommendations\n', recommendations, file=sys.stderr)

        return render_template('recommended_movies.html', len=len(recommendations), recommendations=recommendations)

        #######################

    return render_template('home_guest.html', len=len(MOVIE_GENRES), MOVIE_GENRES=MOVIE_GENRES)

#######################################################

# 404 Page
# @app.route('/<page_name>')
# def other_page(page_name):
#     response = make_response('The page named %s does not exist.' \
#                             % page_name, 404)
#     return response

@app.errorhandler(404)
def page_not_found(error):
    return 'Page Not Found 404'

################################################ Functions ################################################


######################## Webcam + Face Detection Function ########################

def take_photo_detect_face(photo_path):
    # Load the file to detect the faces
    faceCascade = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    # cascPath = sys.argv[1]
    # faceCascade = cv2.CascadeClassifier(cascPath)

    # Open the camera
    video_capture = cv2.VideoCapture(0)

    # Used in the while loop
    faces = ()
    face_images = []

    # If no face was detected or there is more than one face detected loop and don't capture photo 
    while faces == () or len(face_images) > 1:
        # Capture frame-by-frame & convert to gray scale
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # minimize the frame
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        face_images= []
        
        # Draw a rectangle around the face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
            # Get the area inside the rectangle
            face_images.append( frame[y+2:y+h-1, x+2:x+w-1] )
            # cv2.imshow( frame[y:y+h, x:x+w] )

        # Display the resulting frame
        cv2.imshow('Camera', face_images[0])
                
        # print(face_images)

    # Save the frame
    cv2.imwrite(photo_path, face_images[0])

    # When everything is done, release the capture and close all opened windows
    video_capture.release()
    cv2.destroyAllWindows()

######################## Webcam Function ########################

def take_photo_timer(photo_path):
    # SET THE COUNTDOWN TIMER
    TIMER = int(5)

    # Open the camera
    cap = cv2.VideoCapture(0)

    # begin countdown
    prev = time.time()

    # Keep looping until photo is taken
    while TIMER >= 0:
        # Read and display each frame
        ret, img = cap.read()

        # Display countdown on each frame
        # specify the font and draw the countdown using putText
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, str(TIMER), (200, 250), font, 7, (255, 255, 255), 7, cv2.LINE_AA)
        cv2.imshow('Camera', img)
        cv2.waitKey(125)

        # current time
        cur = time.time()

        # Update and keep track of Countdown
        # if time elapsed is one second then decrease the counter
        if cur-prev >= 1:
            prev = cur
            TIMER = TIMER-1

    # Read and display each frame
    ret, img = cap.read()

    # Display the clicked frame for 2 sec. Also increased time by 1 sec in waitKey
    cv2.imshow('Photo', img)

    # time for which image displayed
    cv2.waitKey(1000)

    # Save the frame
    cv2.imwrite(photo_path, img)

    # close the camera
    cap.release()

    # close all the opened windows
    cv2.destroyAllWindows()


######################## Camera + Face Detection + Timer Function ########################

def take_photo_timer_detect_face(photo_path):
    # This file is used to detect the faces
    faceCascade = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    
    # Used in the while loop
    faces = ()
    face_images = []

    # Open the camera
    cap = cv2.VideoCapture(0)

    # If no face was detected or there is more than one face detected loop and don't capture photo 
    while faces == () or len(face_images) > 1:
        # Set the countdown timer
        TIMER = int(3)
        
        # begin countdown
        prev = time.time()
        
        # Keep looping until timer is over & photo is taken
        while TIMER >= 0:
            # Reset the countdown timer whenever there is no face or more than a face detected
            if faces == () or len(face_images) > 1:
                TIMER = int(3)
                prev = time.time()
            
            # Read and display each frame & convert to gray scale
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Display countdown on each frame
            # specify the font and draw the countdown using putText
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(frame, 'Show Your Face', (50, 50), font, 2, (255, 50, 50), 3, cv2.LINE_AA)
            cv2.putText(frame, str(TIMER), (50, 150), font, 3, (255, 50, 50), 3, cv2.LINE_AA)
            cv2.imshow('Camera', frame)
            cv2.waitKey(150)
            
            # find faces and reduce the image scale
            # returns the positions of detected faces as Rect(x,y,w,h).
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            face_images= []
            
            # Draw a rectangle around the face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Get the area inside the rectangle
                face_images.append( frame[y+2:y+h-1, x+2:x+w-1] )
                
            # current time
            cur = time.time()
            
            # Update and keep track of Countdown
            # if time elapsed is one second then decrease the counter
            if cur-prev >= 1:
                prev = cur
                TIMER = TIMER-1
   
    # Read and display each frame
    ret, img = cap.read()

    # Display the clicked frame for 2 sec. Also increased time by 1 sec in waitKey
    cv2.imshow('Photo', face_images[0])

    # time for which image displayed
    cv2.waitKey(1000)

    # Save the frame
    cv2.imwrite(photo_path, face_images[0])

    # close the camera
    cap.release()

    # close all the opened windows
    cv2.destroyAllWindows()
    

######################## Model Function ########################

# Predict either age, ethnicity or gender prediction
def model_pred(MODEL_PATH, MODEL_LABELS, photo_path):
    # Load the user photo & resize it
    img = cv2.imread(photo_path)
    # img = cv2.imread("static/user_photos/europeanPerson3.png")
    resized_img = cv2.resize(img, (48, 48))
    resized_img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    resized_img_gray_shape = (1, 48, 48, 1)
    resized_img_gray = resized_img_gray.reshape(*resized_img_gray_shape)
    X = resized_img_gray

    # Load the model
    model = load_model(MODEL_PATH)
    pred = model.predict(X)
    
    # Selecting the model label that was predicted
    y_model_predict = np.argmax(pred,axis=1)
    prediction = MODEL_LABELS[int(y_model_predict)]

    return prediction

######################## Email Functions ########################

# Send Email Verfication
def verify_mail(email, otp):
    user_email = email

    msg = Message('Flask APP verification', sender = 'recommenderteam6@gmail.com', recipients = [user_email])  
    msg.body = 'Your verification Code\n' + str(otp)  
    mail.send(msg)

    return 'Sent'


# Send mail to user
def send_recommendations_to_mail(user_info, movie_genres, recommendations):
    user_fname = user_info[1]
    user_email = user_info[3]

    recommendations_str = ""
    for movie in recommendations:
        recommendations_str += str(movie)+'\n'

    msg = Message('Movie Recommendations from Flask APP', sender ='recommenderteam6@gmail.com', recipients = [user_email] )
    msg.body = 'Hello ' + user_fname + ',' + '\n' + \
    'Chosen Movie Genre : "' + movie_genres + '"\n' + \
    'Here are some movie recommendations for you :' + '\n' + \
    recommendations_str + '\n' + \
    'Enjoy '+ emoji.emojize(":grinning_face_with_big_eyes:")
    mail.send(msg)
    return 'Sent'


######################## Recommender Function ########################
def recommend_movies(n_top, age, gender, genre, df_ratings_final, df_size, df_movies):
    # Loop on all movie genres and
    exists=[]
    for row in df_ratings_final.movie_genres:
        for item in genre:
            if row.find(item):
                found=1
                exists.append(found)
                break
                
                
    df_new=df_ratings_final.copy()
    
    df_new['exists']=exists
    
    df_new=df_new[(df_new['bucketized_user_age']==age) & (df_new['exists']==1) & (df_new['user_gender']==gender)]
    df_new=df_new.groupby('movie_id').mean('user_rating')
    
    #df_new=pd.concat([df_new, df_size],  axis=1, join = 'outer')
    df_new=pd.merge(df_new, df_size, left_index=True, right_index=True, how='left')
    df_new=df_new.sort_values(by=['user_rating','year','size'],ascending=False)
    final_df=pd.merge(df_new,df_movies, left_index=True, right_index=True, how='left')
    
    recommended_movies=list(final_df['movie_title'])[0:n_top]
    
    return recommended_movies


######################## RUN APP ########################

# test = get_all_users()
# print('test', test)

#  __name__ gets the value "__main__" when executing the script
# Run when the script is executed
if __name__ == '__main__':
    app.run(debug=True)