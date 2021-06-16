const express = require('express')
const multer = require('multer')
const morgan = require('morgan')
const path = require('path')

const app = express()

//port
const port = 3000 || process.env.port

//ejs view engine 
app.set('view engine', 'ejs')

//static files
app.use(express.static(__dirname + "/public") )
app.use(morgan('dev'))

//define storage for images
let storage = multer.diskStorage({
    destination:(req, file, cb)=>{
        cb(null,'./src/data/images');
    },
    filename:(req, file, cb) =>{
        cb(null,  req.body.image.name + path.extname(file.originalname))
    }
});

const upload = multer({storage : storage});


//get images
app.get('/', (req, res) => {
    res.render('index')
})

//post image
app.post('/', upload.single('image'), (req, res) => {
    Image.create({filename: req.file.filename}, (err) =>{
        if (err) return next(err);
        res.redirect('/');
    })
})

//listen to the server
app.listen(port, () => {
    console.log(`Listening to the server at port: ${port}`)
})