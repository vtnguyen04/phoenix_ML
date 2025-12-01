import React from 'react';

const Header: React.FC = () => {
  return (
    <header style={{ padding: '1rem', background: '#333', color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <h2>Vehicle Monitoring Platform</h2>
      <div>User: Admin</div>
    </header>
  );
};

export default Header;
