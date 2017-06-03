function varargout = app(varargin)
% APP MATLAB code for app.fig
%      APP, by itself, creates a new APP or raises the existing
%      singleton*.
%
%      H = APP returns the handle to a new APP or the handle to
%      the existing singleton*.
%
%      APP('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in APP.M with the given input arguments.
%
%      APP('Property','Value',...) creates a new APP or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before app_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to app_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help app

% Last Modified by GUIDE v2.5 13-Apr-2017 16:52:00

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @app_OpeningFcn, ...
                   'gui_OutputFcn',  @app_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before app is made visible.
function app_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to app (see VARARGIN)

% Choose default command line output for app
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% share the image resource
setappdata(handles.figure_app, 'img_src', 0);
setappdata(handles.figure_app, 'Theta1', 0);
setappdata(handles.figure_app, 'Theta2', 0);

% UIWAIT makes app wait for user response (see UIRESUME)
% uiwait(handles.figure_app);


% --- Outputs from this function are returned to the command line.
function varargout = app_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --------------------------------------------------------------------
function m_file_Callback(hObject, eventdata, handles)
% hObject    handle to m_file (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function m_file_open_Callback(hObject, eventdata, handles)
% hObject    handle to m_file_open (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename, pathname] = uigetfile(...
        {'*.bmp; *.jpg; *.png; *.jpeg; *.tiff',...
        'Image Files (*.bmp;*.jpg; *.png; *.jpeg; *.tiff)';...
        '*.*', 'All Files (*.*)'},...
        'Pick an image');
if isequal(filename, 0) || isequal(pathname, 0)
    return;
end
axes(handles.axes_src);    % operate axes_sev
fpath = [pathname filename]; % add the filename and pathname together to locate the image
img_src = imread(fpath);
imshow(img_src);       % read the file with 'imread', and display the image with imshow
setappdata(handles.figure_app, 'img_src', img_src);
    

% --------------------------------------------------------------------
function m_file_save_Callback(hObject, eventdata, handles)
% hObject    handle to m_file_save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
img_src = getappdata(handles.figure_app, 'img_src');
[filename, pathname] = uiputfile({'*.bmp', 'BMP files';...
    '*.jpg', 'JPG files'; '*.jpeg', 'JPEG files';...
    '*.tiff', 'TIFF files'}, 'Pick an Image');
if isequal(filename, 0) || isequal(pathname, 0)
    reture;  % if cancel
else
    fpath = fullfile(pathname, filename);
end
imwrite(img_src, fpath);


% --------------------------------------------------------------------
function m_file_exit_Callback(hObject, eventdata, handles)
% hObject    handle to m_file_exit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
delete(handles.figure_app);


% --------------------------------------------------------------------
function m_para_Callback(hObject, eventdata, handles)
% hObject    handle to m_para (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename, pathname] = uigetfile(...
        {'*.mat'},...
        'Choose a parameteres set');
if isequal(filename, 0) || isequal(pathname, 0)
    return;
end
fpath = [pathname filename];
load(fpath);
setappdata(handles.figure_app, 'Theta1', Theta1);  % set Theta1
setappdata(handles.figure_app, 'Theta2', Theta2);  % set Theta2
pred_Test = predict(Theta1, Theta2, X_test);
[b, y_Test] = max(y_test,[],2);
acc =mean(double(pred_Test == y_Test)) * 100;
set(handles.text_acc, 'String', [num2str(acc) '%']);


% --- Executes on button press in pushbutton_rec.
function pushbutton_rec_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_rec (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
theta1 = getappdata(handles.figure_app, 'Theta1');
theta2 = getappdata(handles.figure_app, 'Theta2');
picture = getappdata(handles.figure_app, 'picture');
img_src = getappdata(handles.figure_app, 'img_src');
if ~isempty(picture)
    img_lbp = extractLBPFeatures(picture, 'CellSize', [16 16]);
    img_lbp = mapminmax(img_lbp, 0, 0.5);
    p = predict(theta1, theta2, img_lbp);

elseif ~isempty(img_src)
    img_lbp = extractLBPFeatures(img_src, 'CellSize', [16 16]);
    img_lbp = mapminmax(img_lbp, 0, 0.5);
    p = predict(theta1, theta2, img_lbp);
end

% img_lbp = extractLBPFeatures(img_src, 'CellSize', [16 16]);
% img_lbp = mapminmax(img_lbp, 0, 0.5);
% p = predict(theta1, theta2, img_lbp);
switch  p
        case 1
            text_content = 'Angry';
        case 2
            text_content = 'Disgust';
        case 3
            text_content = 'Fear';
        case 4
            text_content = 'Happy';
        case 5
            text_content = 'Neutral';
        case 6
            text_content = 'Sad';
    otherwise
            text_content = 'Surprise';
end
set(handles.text_exp, 'String', text_content);


% --- Executes on button press in pushbutton_cam.
function pushbutton_cam_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_cam (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global vid1
vid1 = videoinput('macvideo', 1);
img1_res = get(vid1, 'videoResolution');
nBand = get(vid1, 'NumberofBands');   % color channels
axes(handles.axes_src);
hImage1 = imshow(zeros(img1_res(2), img1_res(1), nBand));
preview(vid1,hImage1);





%% ================= Tracking the face in a video =========================
% Get the skin tone information by extracting the Hue from the video frame
% converted to the HSV color space.
% [hueChannel,~,~] = rgb2hsv(frame);
% 
% % Display the Hue Channel data and draw the bounding box around the face.
% figure, imshow(hueChannel), title('Hue channel data');
% rectangle('Position',bbox(1,:),'LineWidth',2,'EdgeColor',[1 1 0])
% 
% % Detect the nose within the face region. The nose provides a more accurate
% % measure of the skin tone because it does not contain any background
% % pixels.
% noseDetector = vision.CascadeObjectDetector('Nose', 'UseROI', true);
% noseBBox     = step(noseDetector, frame, bbox(1,:));
% 
% % Create a tracker object.
% tracker = vision.HistogramBasedTracker;
% 
% % Initialize the tracker histogram using the Hue channel pixels from the
% % nose.
% initializeObject(tracker, hueChannel, noseBBox(1,:));
% 
% % Create a video player object for displaying video frames.
% % videoInfo    = info(videoFileReader);
% videoPlayer  = vision.VideoPlayer();%'Position',[300 300 videoInfo.VideoSize+30]);
% 
% % Track the face over successive video frames until the video is finished.
% while 1
% 
%     % Extract the next video frame
%     videoFrame = getsnapshot(vid1);
% 
%     % RGB -> HSV
%     [hueChannel,~,~] = rgb2hsv(videoFrame);
% 
%     % Track using the Hue channel data
%     bbox = step(tracker, hueChannel);
% 
%     % Insert a bounding box around the object being tracked
%     videoOut = insertObjectAnnotation(videoFrame,'rectangle',bbox,'Face');
% 
%     % Display the annotated video frame using the video player object
%     step(videoPlayer, videoOut);
% 
% end


% --- Executes on button press in pushbutton_shot.
function pushbutton_shot_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_shot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global vid1
% frame = getsnapshot(vid1);
% frame = ycbcr2rgb(frame);
% axes(handles.axes_src);
% imshow(frame);
%% ======================== Detect ========================================
Detector = vision.CascadeObjectDetector('UpperBody');
Detector.MinSize = [200 200];
% Detector.MergeThreshold = 1;

% pause(1);  % wait for 1 second
frame = getsnapshot(vid1); % get a frame from vid1
frame = ycbcr2rgb(frame);  
% frame = rgb2gray(frame);   % change rgb image to gray picture
% axes(handles.VideoProcessing);
% imshow(frame,[]);  
% Read a video frame and run the detector.
bbox = step(Detector, frame);

% Draw the returned bounding box around the detected face.
I = insertObjectAnnotation(frame, 'rectangle', bbox, 'face');
axes(handles.axes_src), imshow(I), title('Detected face');
pic_cropped = imcrop(I, bbox);
pic_cropped = rgb2gray(pic_cropped);
pic_cropped = cut(pic_cropped);
picture = imresize(pic_cropped, [256 256]);
figure, imshow(picture);
setappdata(handles.figure_app, 'picture', picture);



