import logo from './logo.svg';
import './App.css';
import PromptForm from './PromptForm';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <h1>
          Teaching Assistant
        </h1>
        <p>Ask me anything...</p>
        <PromptForm />
        
      </header>
    </div>
  );
}

export default App;
