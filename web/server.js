const express = require('express')
const multer = require('multer')
const morgan = require('morgan')
const path = require('path')
const crypto = require('crypto')
const mongoose = require('mongoose')
const {GridFsStorage} = require('multer-gridfs-storage')
const Grid = require('gridfs-stream')

const app = express()

//port
const port = process.env.PORT || 3000



//ejs view engine 
app.set('view engine', 'ejs')
app.use(express.json())
app.use(morgan('dev'))


//mongodb uri
const mongoURI = process.env.MONGODB_URI 

//mongodb connection
const conn = mongoose.createConnection(mongoURI, {useNewUrlParser: true, useUnifiedTopology: true}, () =>
    { console.log("Database connected!!!")}
);

let gfs;
let filename;


conn.once('open',  () => {
    gfs = Grid(conn.db, mongoose.mongo);
    gfs.collection('uploads');
  })

  const storage = new GridFsStorage({
    url: mongoURI,
    file: (req, file) => {
      return new Promise((resolve, reject) => {
        crypto.randomBytes(16, (err, buf) => {
          if (err) {
            return reject(err);
          }
        filename = buf.toString('hex') + path.extname(file.originalname);
          const fileInfo = {
            filename: filename,
            bucketName: 'uploads'
          };
          resolve(fileInfo);
        });
      });
    }
  });
  const upload = multer({ storage });


//get images
app.get('/', (req, res) => {
    gfs.files.find().sort({uploadDate: -1}).toArray((err, files) => {
        if (!files || files.length == 0){
            res.render('index', {files: false})
        }else{
            files.map(file => {
                if(file.contentType === 'image/jpeg' || file.contentType === 'image/png')
                {
                   file.isImage = true;
                } else{
                    file.isImage = false;
                }
            });
            res.render('index', {files: files})
        }
    })
})

//post image
app.post('/upload', upload.single('file'), (req, res) => {
   res.redirect('/')
})

//get image filename as json
app.get('/img', (req, res) => {
    res.json({filename})
})

//list of files
app.get('/files', (req, res) => {
    gfs.files.find().toArray((err, files) => {
        if (!files || files.length == 0){
            res.status(404).json({
                err: "No images exists"
            });
        }
        return res.json(files);
    })
})

app.get('/file', (req, res) => {
    gfs.files.findOne({filename :req.file.filename}, (err, file) => {
        if (!file|| file.length == 0){
            res.status(404).json({
                err: "No image exists"
            });
        }
        return res.json(file);
    });

})

app.get('/image/:filename', (req, res) => {
    gfs.files.findOne({filename: req.params.filename}, (err, file) => {
        if (!file|| file.length == 0){
            res.status(404).json({
                err: "No image exists"
            });
        }
        if(file.contentType === 'image/jpeg' || file.contentType === 'image/png'){
            const readstream = gfs.createReadStream(file.filename);
            readstream.pipe(res);
        } else{
            res.status(404).json({
                err: "Not an image"
            })
        }
    });

})



//listen to the server
app.listen(port, () => {
    console.log(`Listening to the server at port: ${port}`)
})
