import React from "react";
import ImageUploader from "react-images-upload";

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = { pictures: [] };
    this.onDrop = this.onDrop.bind(this);
  }

  onDrop(pictureFiles, pictureDataURLs) {
    this.setState({
      pictures: this.state.pictures.concat(pictureFiles)
    });
    // 여기에 http 요청 작성
  }

  render() {
    return (
      <ImageUploader
        withIcon={true}
        buttonText="이미지를 선택하세요"
        onChange={this.onDrop}
        imgExtension={[".jpg", ".gif", ".png", ".gif"]}
        maxFileSize={5242880}
        withPreview={true}
      />
    );
  }
}

export default App;