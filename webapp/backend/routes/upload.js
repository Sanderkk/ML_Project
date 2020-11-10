var express = require('express');
let multer = require('multer');
let path = require('path');
let helpers = require('../helpers');
let tf = require('@tensorflow/tfjs-node');
let fs = require('fs'); 
//require('@tensorflow/tfjs-node')

const readImage = relPath => {
    const imageBuffer = fs.readFileSync(path.join(__dirname,relPath));
    const tfimage = tf.node.decodeImage(imageBuffer); //default #channel 4
    
    const smalImg = tf.image.resizeBilinear(tfimage, [224, 224]);
    const resized = tf.cast(smalImg, 'float32');
    const t4d = tf.tensor4d(Array.from(resized.dataSync()),[1,224,224,3])
    return t4d;
};

//This is if you convert the model to json first
//const model = tf.loadGraphModel('file://trent_model/ny_CNN_model/model.json').then(function(model){predictFunc(model)});
//model.then(function(){console.log("LOADED")})


//let model = tf.node.loadSavedModel('./trent_model/CNN_Model/').then(function(model){predictFunc(model)});

let model = undefined;
let result = undefined;

function startLoadModel() {
    return loadModel();
}
function startPredictModel(pathString) {
    return predictModel(pathString);
}

async function loadModel() {
    return tf.node.loadSavedModel('./trent_model/CNN_Model/') //.then(function(model){predictFunc(model)});
}
async function predictModel(pathString) {
    return model.predict(readImage(pathString));
}


var app = express.Router();
app.use(express.static('uploads'));

const storage = multer.diskStorage({
  destination: function(req, file, cb) {
      cb(null, 'uploads/');
  },

  // By default, multer removes file extensions so let's add them back
  filename: function(req, file, cb) {
      cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname));
  }
});

//app.listen(port, () => console.log(`Listening on port ${port}...`));

app.post('/', (req, res) => {
  // 'profile_pic' is the name of our file input field in the HTML form
  let upload = multer({ storage: storage, fileFilter: helpers.imageFilter }).single('profile_pic');

  upload(req, res, function(err) {
      // req.file contains information of uploaded file
      // req.body contains information of text fields, if there were any

      if (req.fileValidationError) {
          return res.send(req.fileValidationError);
      }
      //else if (!req.file) {
      //    return res.send('Please select an image to upload');
      //}
      else if (err instanceof multer.MulterError) {
          return res.send(err);
      }
      else if (err) {
          return res.send(err);
      }

      console.log('IMAGE UPLOADED.');

      (async() => {
        console.log('LOADING MODEL');
        model = await startLoadModel();
        //result = await startPredictModel('../trent_model/testing/n02085620-Chihuahua/n02085620_9414.jpg');
        console.log('LOADING MODEL DONE. PREDICTING WITH IMG ' + req.file.path);
        result = await startPredictModel('../'+req.file.path);
        console.log(result.arraySync());
        let resultArray = result.arraySync();
        let myJSON = {};
        myJSON["path"] = req.file.path;
        myJSON["values"] = resultArray[0];
        //resultArray[0].map((value,index) => {myJSON[index] = value});
        console.log('PREDICT MODEL DONE. SENDING...');
        //res.send(`You have uploaded this image: <hr/><img src="../${req.file.path}" width="500"><hr /><a href="./">Upload another image</a>`);
        res.send(myJSON);
      })();

  });
});

app.get('/', (req, res) => {
    res.send('Hello World!')
})


module.exports = app;