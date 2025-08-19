import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.ui_components import apply_custom_css, create_metric_card, create_info_card
    from utils.auth import check_authentication
    apply_custom_css()
except ImportError:
    def create_metric_card(title, value, delta=None):
        return st.metric(title, value, delta)
    def create_info_card(title, content):
        return st.info(f"**{title}**\n\n{content}")
    def check_authentication():
        return True, None

class InteractiveBacktestVisualizer:
    """Advanced visualization system for backtesting results"""
    
    def __init__(self):
        self.colors = {
            'primary': '#00D4FF',
            'secondary': '#FF6B6B', 
            'success': '#4ECDC4',
            'warning': '#FFA726',
            'danger': '#F44336',
            'info': '#42A5F5',
            'background': 'rgba(0,0,0,0)',
            'grid': 'rgba(128,128,128,0.2)'
        }
    
    def create_performance_dashboard(self, results_dict, metrics_dict, symbol):
        """Create comprehensive performance dashboard"""
        
        # Performance Overview Cards
        st.markdown("### 📊 Performance Overview")
        
        if results_dict:
            # Calculate best performing strategy
            best_strategy = max(metrics_dict.keys(), 
                              key=lambda x: float(metrics_dict[x]['Annual Return'].replace('%', '')) / 100)
            best_metrics = metrics_dict[best_strategy]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                create_metric_card(
                    "Best Strategy", 
                    best_strategy,
                    f"{best_metrics['Annual Return']}"
                )
            
            with col2:
                create_metric_card(
                    "Max Annual Return",
                    best_metrics['Annual Return'],
                    f"Sharpe: {best_metrics['Sharpe Ratio']}"
                )
            
            with col3:
                create_metric_card(
                    "Best Sharpe Ratio",
                    best_metrics['Sharpe Ratio'],
                    f"Max DD: {best_metrics['Max Drawdown']}"
                )
            
            with col4:
                create_metric_card(
                    "Win Rate",
                    best_metrics['Win Rate'],
                    f"Trades: {best_metrics['Total Trades']}"
                )
        
        # Interactive Performance Chart
        st.markdown("### 📈 Interactive Performance Analysis")
        
        fig = self.create_interactive_performance_chart(results_dict)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk-Return Scatter Plot
        st.markdown("### 🎯 Risk-Return Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_return_fig = self.create_risk_return_scatter(metrics_dict)
            st.plotly_chart(risk_return_fig, use_container_width=True)
        
        with col2:
            sharpe_comparison = self.create_sharpe_comparison(metrics_dict)
            st.plotly_chart(sharpe_comparison, use_container_width=True)
        
        # Drawdown Analysis
        st.markdown("### 📉 Drawdown Analysis")
        
        drawdown_fig = self.create_drawdown_heatmap(results_dict)
        st.plotly_chart(drawdown_fig, use_container_width=True)
        
        # Rolling Performance Metrics
        st.markdown("### 🔄 Rolling Performance Metrics")
        
        rolling_fig = self.create_rolling_metrics_chart(results_dict)
        st.plotly_chart(rolling_fig, use_container_width=True)
        
        # Distribution Analysis
        st.markdown("### 📊 Returns Distribution Analysis")
        
        dist_fig = self.create_returns_distribution(results_dict)
        st.plotly_chart(dist_fig, use_container_width=True)
        
        # Correlation Matrix
        st.markdown("### 🔗 Strategy Correlation Matrix")
        
        corr_fig = self.create_correlation_heatmap(results_dict)
        st.plotly_chart(corr_fig, use_container_width=True)
    
    def create_interactive_performance_chart(self, results_dict):
        """Create interactive performance chart with multiple views"""
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Cumulative Returns', 'Daily Returns', 'Rolling Volatility'),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['success'], 
                 self.colors['warning'], self.colors['info']]
        
        for i, (name, data) in enumerate(results_dict.items()):
            color = colors[i % len(colors)]
            
            # Cumulative returns
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=(data['strategy_returns'].fillna(0).cumsum() * 100),
                    mode='lines',
                    name=f'{name}',
                    line=dict(color=color, width=2),
                    hovertemplate=f'{name}<br>Date: %{{x}}<br>Cumulative Return: %{{y:.2f}}%<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Daily returns
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=(data['strategy_returns'].fillna(0) * 100),
                    mode='lines',
                    name=f'{name} Daily',
                    line=dict(color=color, width=1),
                    showlegend=False,
                    hovertemplate=f'{name}<br>Date: %{{x}}<br>Daily Return: %{{y:.2f}}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Rolling volatility (30-day)
            rolling_vol = data['strategy_returns'].rolling(30).std() * np.sqrt(252) * 100
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=rolling_vol,
                    mode='lines',
                    name=f'{name} Vol',
                    line=dict(color=color, width=1),
                    showlegend=False,
                    hovertemplate=f'{name}<br>Date: %{{x}}<br>30-Day Vol: %{{y:.2f}}%<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Add buy & hold benchmark if available
        if results_dict:
            first_result = list(results_dict.values())[0]
            fig.add_trace(
                go.Scatter(
                    x=first_result.index,
                    y=(first_result['returns'].fillna(0).cumsum() * 100),
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='#FFA726', width=2, dash='dash'),
                    hovertemplate='Buy & Hold<br>Date: %{x}<br>Cumulative Return: %{y:.2f}%<extra></extra>'
                ),
                row=1, col=1
            )
        
        fig.update_layout(
            title='Interactive Performance Analysis',
            height=800,
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color='white'),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(gridcolor=self.colors['grid'], zeroline=False)
        fig.update_yaxes(gridcolor=self.colors['grid'], zeroline=False)
        
        # Add range selector
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        return fig
    
    def create_risk_return_scatter(self, metrics_dict):
        """Create risk-return scatter plot"""
        
        strategies = list(metrics_dict.keys())
        returns = [float(metrics_dict[s]['Annual Return'].replace('%', '')) for s in strategies]
        volatilities = [float(metrics_dict[s]['Annual Volatility'].replace('%', '')) for s in strategies]
        sharpe_ratios = [float(metrics_dict[s]['Sharpe Ratio']) for s in strategies]
        
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=volatilities,
            y=returns,
            mode='markers+text',
            text=strategies,
            textposition="top center",
            marker=dict(
                size=[abs(sr) * 10 + 10 for sr in sharpe_ratios],  # Size based on Sharpe ratio
                color=sharpe_ratios,
                colorscale='RdYlBu',
                colorbar=dict(title="Sharpe Ratio"),
                showscale=True,
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         'Annual Return: %{y:.1f}%<br>' +
                         'Volatility: %{x:.1f}%<br>' +
                         'Sharpe Ratio: %{marker.color:.2f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title='Risk-Return Profile by Strategy',
            xaxis_title='Annual Volatility (%)',
            yaxis_title='Annual Return (%)',
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color='white'),
            height=400
        )
        
        fig.update_xaxes(gridcolor=self.colors['grid'], zeroline=False)
        fig.update_yaxes(gridcolor=self.colors['grid'], zeroline=False)
        
        return fig
    
    def create_sharpe_comparison(self, metrics_dict):
        """Create Sharpe ratio comparison chart"""
        
        strategies = list(metrics_dict.keys())
        sharpe_ratios = [float(metrics_dict[s]['Sharpe Ratio']) for s in strategies]
        
        # Color code based on Sharpe ratio quality
        colors = []
        for sr in sharpe_ratios:
            if sr > 1.5:
                colors.append(self.colors['success'])  # Excellent
            elif sr > 1.0:
                colors.append(self.colors['primary'])  # Good
            elif sr > 0.5:
                colors.append(self.colors['warning'])  # Fair
            else:
                colors.append(self.colors['danger'])   # Poor
        
        fig = go.Figure(data=[
            go.Bar(
                x=strategies,
                y=sharpe_ratios,
                marker_color=colors,
                text=[f'{sr:.2f}' for sr in sharpe_ratios],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Sharpe Ratio: %{y:.2f}<extra></extra>'
            )
        ])
        
        # Add quality zones
        fig.add_hline(y=1.0, line_dash="dash", line_color="white", 
                     annotation_text="Good (>1.0)", annotation_position="left")
        fig.add_hline(y=1.5, line_dash="dash", line_color="green", 
                     annotation_text="Excellent (>1.5)", annotation_position="left")
        
        fig.update_layout(
            title='Sharpe Ratio Comparison',
            xaxis_title='Strategy',
            yaxis_title='Sharpe Ratio',
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color='white'),
            height=400
        )
        
        fig.update_xaxes(gridcolor=self.colors['grid'], zeroline=False)
        fig.update_yaxes(gridcolor=self.colors['grid'], zeroline=False)
        
        return fig
    
    def create_drawdown_heatmap(self, results_dict):
        """Create drawdown analysis heatmap"""
        
        fig = make_subplots(rows=len(results_dict), cols=1,
                           subplot_titles=list(results_dict.keys()),
                           vertical_spacing=0.05)
        
        for i, (name, data) in enumerate(results_dict.items(), 1):
            # Calculate drawdown
            strategy_returns = data['strategy_returns'].fillna(0)
            cumulative = (1 + strategy_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max * 100
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=drawdown,
                    mode='lines',
                    fill='tonexty' if i == 1 else 'tozeroy',
                    name=f'{name} DD',
                    line=dict(color=self.colors['danger'], width=1),
                    fillcolor=f'rgba(244, 67, 54, 0.3)',
                    hovertemplate=f'{name}<br>Date: %{{x}}<br>Drawdown: %{{y:.2f}}%<extra></extra>'
                ),
                row=i, col=1
            )
        
        fig.update_layout(
            title='Strategy Drawdown Analysis',
            height=150 * len(results_dict),
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color='white'),
            showlegend=False
        )
        
        fig.update_xaxes(gridcolor=self.colors['grid'], zeroline=False)
        fig.update_yaxes(gridcolor=self.colors['grid'], zeroline=False, title_text="Drawdown (%)")
        
        return fig
    
    def create_rolling_metrics_chart(self, results_dict):
        """Create rolling performance metrics chart"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Rolling Sharpe Ratio (30D)', 'Rolling Max Drawdown (30D)', 
                           'Rolling Win Rate (30D)', 'Rolling Volatility (30D)'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['success'], 
                 self.colors['warning']]
        
        for i, (name, data) in enumerate(results_dict.items()):
            color = colors[i % len(colors)]
            returns = data['strategy_returns'].fillna(0)
            
            # Rolling Sharpe Ratio (30-day)
            rolling_sharpe = (returns.rolling(30).mean() * 252) / (returns.rolling(30).std() * np.sqrt(252))
            fig.add_trace(
                go.Scatter(x=data.index, y=rolling_sharpe, mode='lines', name=f'{name}',
                          line=dict(color=color, width=2), showlegend=True,
                          hovertemplate=f'{name}<br>Date: %{{x}}<br>30D Sharpe: %{{y:.2f}}<extra></extra>'),
                row=1, col=1
            )
            
            # Rolling Max Drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.rolling(30).max()
            rolling_dd = ((cumulative - rolling_max) / rolling_max * 100).rolling(30).min()
            fig.add_trace(
                go.Scatter(x=data.index, y=rolling_dd, mode='lines', name=f'{name}',
                          line=dict(color=color, width=2), showlegend=False,
                          hovertemplate=f'{name}<br>Date: %{{x}}<br>30D Max DD: %{{y:.2f}}%<extra></extra>'),
                row=1, col=2
            )
            
            # Rolling Win Rate
            rolling_wins = (returns > 0).rolling(30).mean() * 100
            fig.add_trace(
                go.Scatter(x=data.index, y=rolling_wins, mode='lines', name=f'{name}',
                          line=dict(color=color, width=2), showlegend=False,
                          hovertemplate=f'{name}<br>Date: %{{x}}<br>30D Win Rate: %{{y:.1f}}%<extra></extra>'),
                row=2, col=1
            )
            
            # Rolling Volatility
            rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
            fig.add_trace(
                go.Scatter(x=data.index, y=rolling_vol, mode='lines', name=f'{name}',
                          line=dict(color=color, width=2), showlegend=False,
                          hovertemplate=f'{name}<br>Date: %{{x}}<br>30D Volatility: %{{y:.1f}}%<extra></extra>'),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Rolling Performance Metrics (30-Day Window)',
            height=600,
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color='white'),
            hovermode='x unified'
        )
        
        fig.update_xaxes(gridcolor=self.colors['grid'], zeroline=False)
        fig.update_yaxes(gridcolor=self.colors['grid'], zeroline=False)
        
        return fig
    
    def create_returns_distribution(self, results_dict):
        """Create returns distribution analysis"""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Returns Distribution', 'Q-Q Plot vs Normal'),
            horizontal_spacing=0.1
        )
        
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['success'], 
                 self.colors['warning']]
        
        for i, (name, data) in enumerate(results_dict.items()):
            color = colors[i % len(colors)]
            returns = data['strategy_returns'].fillna(0) * 100  # Convert to percentage
            
            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    name=f'{name}',
                    nbinsx=50,
                    opacity=0.7,
                    marker_color=color,
                    hovertemplate=f'{name}<br>Return Range: %{{x}}%<br>Frequency: %{{y}}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Q-Q Plot
            sorted_returns = np.sort(returns.dropna())
            n = len(sorted_returns)
            theoretical_quantiles = np.linspace(0.01, 0.99, n)
            normal_quantiles = np.percentile(np.random.normal(sorted_returns.mean(), 
                                                            sorted_returns.std(), 10000), 
                                           theoretical_quantiles * 100)
            
            fig.add_trace(
                go.Scatter(
                    x=normal_quantiles,
                    y=sorted_returns,
                    mode='markers',
                    name=f'{name} Q-Q',
                    marker=dict(color=color, size=4),
                    showlegend=False,
                    hovertemplate=f'{name}<br>Theoretical: %{{x:.2f}}%<br>Observed: %{{y:.2f}}%<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Add perfect correlation line for Q-Q plot
        if results_dict:
            first_data = list(results_dict.values())[0]
            returns_sample = first_data['strategy_returns'].fillna(0) * 100
            min_val, max_val = returns_sample.quantile([0.05, 0.95])
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Normal',
                    line=dict(color='white', dash='dash', width=2),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Returns Distribution Analysis',
            height=400,
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color='white')
        )
        
        fig.update_xaxes(gridcolor=self.colors['grid'], zeroline=False)
        fig.update_yaxes(gridcolor=self.colors['grid'], zeroline=False)
        
        # Update axis labels
        fig.update_xaxes(title_text="Daily Returns (%)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Normal (%)", row=1, col=2)
        fig.update_yaxes(title_text="Observed Returns (%)", row=1, col=2)
        
        return fig
    
    def create_correlation_heatmap(self, results_dict):
        """Create strategy correlation matrix"""
        
        if len(results_dict) < 2:
            return go.Figure().add_annotation(
                text="Need at least 2 strategies for correlation analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16, font_color="white"
            )
        
        # Create returns matrix
        returns_df = pd.DataFrame()
        for name, data in results_dict.items():
            returns_df[name] = data['strategy_returns'].fillna(0)
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 12, "color": "white"},
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Strategy Return Correlations',
            height=400,
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color='white')
        )
        
        return fig

def main():
    # Check authentication
    is_authenticated, user_info = check_authentication()
    if not is_authenticated:
        st.warning("Please log in to access the Interactive Backtesting Visualization module.")
        return
        
    st.title("📊 Interactive Backtesting Performance Visualization")
    st.markdown("**Advanced visual analytics for comprehensive strategy performance analysis**")
    
    # Check if we have cached backtest results
    if 'backtest_results' not in st.session_state or 'backtest_metrics' not in st.session_state:
        st.info("No backtesting results found. Please run a backtest first using the Strategy Backtesting module.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🚀 Go to Strategy Backtesting", type="primary", use_container_width=True):
                st.switch_page("pages/strategy_backtesting.py")
        
        with col2:
            st.markdown("**Or load sample data:**")
            if st.button("📈 Load Sample Analysis", use_container_width=True):
                # Create sample data for demonstration
                dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
                
                sample_results = {}
                sample_metrics = {}
                
                # Generate sample data for different strategies
                np.random.seed(42)
                
                strategies = ['Moving Average', 'RSI Strategy', 'Bollinger Bands']
                for i, strategy in enumerate(strategies):
                    returns = np.random.normal(0.0008, 0.02, len(dates))  # Daily returns
                    
                    sample_data = pd.DataFrame({
                        'strategy_returns': returns,
                        'returns': np.random.normal(0.0005, 0.018, len(dates))  # Market returns
                    }, index=dates)
                    
                    sample_results[strategy] = sample_data
                    
                    # Calculate sample metrics
                    annual_return = (1 + returns).prod() ** (252/len(returns)) - 1
                    annual_vol = returns.std() * np.sqrt(252)
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    
                    sample_metrics[strategy] = {
                        'Annual Return': f"{annual_return:.2%}",
                        'Annual Volatility': f"{annual_vol:.2%}",
                        'Sharpe Ratio': f"{sharpe:.2f}",
                        'Max Drawdown': f"{np.random.uniform(-0.15, -0.05):.2%}",
                        'Win Rate': f"{np.random.uniform(0.45, 0.60):.2%}",
                        'Total Trades': str(np.random.randint(80, 150))
                    }
                
                st.session_state.backtest_results = sample_results
                st.session_state.backtest_metrics = sample_metrics
                st.session_state.backtest_symbol = 'SAMPLE'
                st.rerun()
        
        return
    
    # Load results from session state
    results_dict = st.session_state.backtest_results
    metrics_dict = st.session_state.backtest_metrics
    symbol = st.session_state.get('backtest_symbol', 'Unknown')
    
    st.success(f"✅ Loaded backtest results for {symbol} - {len(results_dict)} strategies analyzed")
    
    # Sidebar controls for customization
    with st.sidebar:
        st.header("🎛️ Visualization Controls")
        
        # Strategy selection
        available_strategies = list(results_dict.keys())
        selected_strategies = st.multiselect(
            "Select Strategies to Display",
            available_strategies,
            default=available_strategies,
            help="Choose which strategies to include in the analysis"
        )
        
        if not selected_strategies:
            st.warning("Please select at least one strategy")
            return
        
        # Filter results based on selection
        filtered_results = {k: v for k, v in results_dict.items() if k in selected_strategies}
        filtered_metrics = {k: v for k, v in metrics_dict.items() if k in selected_strategies}
        
        # Chart customization
        st.subheader("Chart Options")
        show_benchmarks = st.checkbox("Show Buy & Hold Benchmark", value=True)
        chart_height = st.slider("Chart Height", 400, 1000, 600, step=50)
        
        # Clear results button
        st.markdown("---")
        if st.button("🗑️ Clear Results", type="secondary"):
            for key in ['backtest_results', 'backtest_metrics', 'backtest_symbol']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Create visualizer instance
    visualizer = InteractiveBacktestVisualizer()
    
    # Generate comprehensive dashboard
    try:
        visualizer.create_performance_dashboard(filtered_results, filtered_metrics, symbol)
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")
        st.info("This might be due to insufficient data or incompatible data format.")
    
    # Additional Analysis Section
    st.markdown("---")
    st.markdown("### 🔍 Additional Analysis Tools")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Results", use_container_width=True):
            # Create downloadable report
            report_data = {
                'Symbol': symbol,
                'Analysis_Date': datetime.now().strftime('%Y-%m-%d'),
                'Strategies': list(filtered_results.keys()),
                'Metrics': filtered_metrics
            }
            
            st.download_button(
                label="Download Performance Report",
                data=str(report_data),
                file_name=f"backtest_report_{symbol}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("🔄 Compare with Market", use_container_width=True):
            st.info("Market comparison analysis - Feature coming soon!")
    
    with col3:
        if st.button("📈 Advanced Metrics", use_container_width=True):
            st.info("Advanced risk metrics analysis - Feature coming soon!")
    
    # Educational content
    with st.expander("📚 Visualization Guide"):
        st.markdown("""
        **Interactive Performance Analysis:**
        - Use the range selector to zoom into specific time periods
        - Hover over data points for detailed information
        - Click legend items to show/hide strategies
        
        **Risk-Return Analysis:**
        - Bubble size represents Sharpe ratio magnitude
        - Color intensity shows Sharpe ratio quality
        - Optimal strategies appear in top-left quadrant (high return, low risk)
        
        **Rolling Metrics:**
        - Shows how strategy performance evolves over time
        - 30-day rolling window provides insight into stability
        - Look for consistent performers vs volatile strategies
        
        **Distribution Analysis:**
        - Q-Q plots show normality of returns
        - Points close to diagonal line indicate normal distribution
        - Outliers suggest tail risk or exceptional performance periods
        """)

if __name__ == "__main__":
    main()