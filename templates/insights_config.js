// insights-config.js - Comprehensive AI Insights Database
// This file contains all the business intelligence rules and patterns

const INSIGHTS_CONFIG = {
    
    // ========================================
    // NUMERICAL DATA INSIGHTS
    // ========================================
    numerical: {
        
        // Statistical Pattern Analysis
        statistical: {
            highVariability: {
                condition: (stats) => stats.cv > 1.0,
                insight: {
                    type: 'warning',
                    icon: '⚠️',
                    title: 'High Variability Detected',
                    template: 'Coefficient of variation is {cv}%. This indicates significant data spread. Consider: 1) Data quality review, 2) Outlier investigation, 3) Segmentation analysis, 4) Normalization techniques.'
                }
            },
            
            extremeVariability: {
                condition: (stats) => stats.cv > 2.0,
                insight: {
                    type: 'warning',
                    icon: '🚨',
                    title: 'Extreme Variability',
                    template: 'CV of {cv}% suggests potential data quality issues. Immediate actions: 1) Check for data entry errors, 2) Identify and investigate outliers, 3) Consider log transformation, 4) Validate data sources.'
                }
            },
            
            lowVariability: {
                condition: (stats) => stats.cv < 0.1,
                insight: {
                    type: 'insight',
                    icon: '🎯',
                    title: 'Highly Consistent Data',
                    template: 'Very low variability (CV: {cv}%). This suggests: 1) Stable processes, 2) Controlled environment, 3) Potential for predictable outcomes, 4) May indicate lack of diversity in sample.'
                }
            },
            
            rightSkewed: {
                condition: (stats) => stats.skewness > 0.5,
                insight: {
                    type: 'insight',
                    icon: '📈',
                    title: 'Right-Skewed Distribution',
                    template: 'Distribution is right-skewed (skewness: {skewness}). Most values cluster below the mean with some high-value outliers. Consider: 1) Log transformation, 2) Focus on median over mean, 3) Investigate high-value cases.'
                }
            },
            
            leftSkewed: {
                condition: (stats) => stats.skewness < -0.5,
                insight: {
                    type: 'insight',
                    icon: '📉',
                    title: 'Left-Skewed Distribution',
                    template: 'Distribution is left-skewed (skewness: {skewness}). Most values cluster above the mean with some low-value outliers. Consider: 1) Investigate low-value cases, 2) Use median for central tendency, 3) Check for floor effects.'
                }
            },
            
            normalDistribution: {
                condition: (stats) => Math.abs(stats.skewness) < 0.5 && stats.cv > 0.1 && stats.cv < 1.0,
                insight: {
                    type: 'insight',
                    icon: '🔔',
                    title: 'Near-Normal Distribution',
                    template: 'Data appears normally distributed (skewness: {skewness}, CV: {cv}%). This enables: 1) Standard statistical tests, 2) Confidence intervals, 3) Outlier detection using z-scores, 4) Regression analysis.'
                }
            },
            
            bimodalSuspicion: {
                condition: (stats) => stats.cv > 0.8 && Math.abs(stats.skewness) < 0.3,
                insight: {
                    type: 'insight',
                    icon: '🎭',
                    title: 'Possible Bimodal Distribution',
                    template: 'High variability with low skewness suggests potential bimodal distribution. Investigate: 1) Natural groupings in data, 2) Different populations mixed, 3) Seasonal patterns, 4) Consider clustering analysis.'
                }
            },
            
            uniformDistribution: {
                condition: (stats) => Math.abs(stats.skewness) < 0.2 && stats.cv > 0.25 && stats.cv < 0.6,
                insight: {
                    type: 'insight',
                    icon: '📏',
                    title: 'Uniform-like Distribution',
                    template: 'Data shows uniform-like distribution pattern. This might indicate: 1) Random sampling, 2) Artificial data generation, 3) Bounded processes, 4) Equal probability events.'
                }
            }
        },
        
        // Business Context Patterns
        businessContext: {
            // Financial Metrics
            revenue: {
                keywords: ['revenue', 'sales', 'income', 'earnings', 'turnover', 'proceeds'],
                insights: [
                    {
                        condition: (stats) => stats.cv > 0.8,
                        insight: {
                            type: 'action',
                            icon: '💰',
                            title: 'Revenue Segmentation Opportunity',
                            template: 'High revenue variability (CV: {cv}%) indicates diverse customer base. Strategy: 1) Create customer tiers (High: >{high_threshold}, Medium: {med_threshold}-{high_threshold}, Low: <{med_threshold}), 2) Develop tier-specific offerings, 3) Focus retention on high-value customers.'
                        }
                    },
                    {
                        condition: (stats) => stats.cv < 0.3,
                        insight: {
                            type: 'insight',
                            icon: '🎯',
                            title: 'Stable Revenue Pattern',
                            template: 'Consistent revenue pattern (CV: {cv}%) suggests: 1) Subscription-based model working well, 2) Stable customer base, 3) Predictable cash flow, 4) Opportunity for premium upselling above ${high_threshold}.'
                        }
                    },
                    {
                        condition: (stats) => stats.skewness > 1.0,
                        insight: {
                            type: 'action',
                            icon: '🏆',
                            title: 'Identify High-Value Customers',
                            template: 'Revenue is heavily right-skewed - few customers drive significant value. Actions: 1) Identify top {top_percentile}% customers generating >${high_threshold}, 2) Create VIP retention programs, 3) Analyze their characteristics for acquisition targeting.'
                        }
                    }
                ]
            },
            
            // Age Demographics
            age: {
                keywords: ['age', 'years', 'born', 'birthday', 'generation'],
                insights: [
                    {
                        condition: (stats) => stats.mean < 25,
                        insight: {
                            type: 'action',
                            icon: '📱',
                            title: 'Gen Z Audience Strategy',
                            template: 'Young audience (avg: {mean} years). Optimize for: 1) Mobile-first experience, 2) Social media marketing (TikTok, Instagram), 3) Authentic brand messaging, 4) Instant gratification features, 5) Sustainability focus.'
                        }
                    },
                    {
                        condition: (stats) => stats.mean >= 25 && stats.mean < 40,
                        insight: {
                            type: 'action',
                            icon: '💼',
                            title: 'Millennial Engagement Strategy',
                            template: 'Millennial audience (avg: {mean} years). Focus on: 1) Work-life balance solutions, 2) Digital convenience, 3) Value-driven purchases, 4) Experience over products, 5) LinkedIn and Facebook marketing.'
                        }
                    },
                    {
                        condition: (stats) => stats.mean >= 40 && stats.mean < 60,
                        insight: {
                            type: 'action',
                            icon: '🏠',
                            title: 'Gen X Strategy',
                            template: 'Gen X audience (avg: {mean} years). Optimize for: 1) Family-focused solutions, 2) Quality and reliability, 3) Email marketing, 4) Facebook engagement, 5) Traditional customer service channels.'
                        }
                    },
                    {
                        condition: (stats) => stats.mean >= 60,
                        insight: {
                            type: 'action',
                            icon: '📞',
                            title: 'Baby Boomer Strategy',
                            template: 'Mature audience (avg: {mean} years). Prioritize: 1) Personal service, 2) Phone support, 3) Clear, simple interfaces, 4) Trust and security messaging, 5) Traditional media channels.'
                        }
                    },
                    {
                        condition: (stats) => stats.cv > 0.4,
                        insight: {
                            type: 'action',
                            icon: '🎯',
                            title: 'Multi-Generational Approach',
                            template: 'Wide age distribution (CV: {cv}%) requires multi-generational strategy: 1) Segment by age groups, 2) Create age-appropriate messaging, 3) Use diverse communication channels, 4) Offer varying service levels.'
                        }
                    }
                ]
            },
            
            // Performance Scores
            score: {
                keywords: ['score', 'rating', 'performance', 'satisfaction', 'quality', 'grade'],
                insights: [
                    {
                        condition: (stats) => stats.mean > stats.max * 0.8,
                        insight: {
                            type: 'insight',
                            icon: '⭐',
                            title: 'Excellent Performance',
                            template: 'High average score ({mean}/{max}). Strengths: 1) Strong performance baseline, 2) Satisfied customer base, 3) Competitive advantage. Focus: 1) Maintain standards, 2) Use as marketing differentiator, 3) Identify what drives top scores.'
                        }
                    },
                    {
                        condition: (stats) => stats.mean < stats.max * 0.6,
                        insight: {
                            type: 'warning',
                            icon: '📉',
                            title: 'Performance Improvement Needed',
                            template: 'Below-average scores ({mean}/{max}). Immediate actions: 1) Root cause analysis for scores <{improvement_threshold}, 2) Customer feedback collection, 3) Process improvement initiative, 4) Staff training program.'
                        }
                    },
                    {
                        condition: (stats) => stats.cv > 0.3,
                        insight: {
                            type: 'action',
                            icon: '🔍',
                            title: 'Inconsistent Performance',
                            template: 'High score variability (CV: {cv}%) indicates inconsistent experience. Actions: 1) Standardize processes, 2) Identify best practices from high scorers, 3) Additional training for low performers, 4) Quality control measures.'
                        }
                    }
                ]
            },
            
            // Time-based Metrics
            time: {
                keywords: ['time', 'duration', 'minutes', 'hours', 'days', 'tenure', 'session', 'visit'],
                insights: [
                    {
                        condition: (stats, colName) => colName.toLowerCase().includes('session') && stats.mean < 5,
                        insight: {
                            type: 'warning',
                            icon: '⏱️',
                            title: 'Short Session Duration',
                            template: 'Average session time is only {mean} minutes. Engagement strategies: 1) Improve onboarding flow, 2) Add interactive elements, 3) Personalize content, 4) Reduce friction points, 5) A/B test engagement features.'
                        }
                    },
                    {
                        condition: (stats, colName) => colName.toLowerCase().includes('tenure') && stats.mean > 365,
                        insight: {
                            type: 'insight',
                            icon: '🏆',
                            title: 'Strong Customer Retention',
                            template: 'High average tenure ({mean} days) indicates strong retention. Leverage this: 1) Create loyalty programs, 2) Referral incentives, 3) Testimonial campaigns, 4) Identify retention drivers for new customer success.'
                        }
                    },
                    {
                        condition: (stats) => stats.cv > 1.5,
                        insight: {
                            type: 'insight',
                            icon: '📊',
                            title: 'Diverse Time Patterns',
                            template: 'High time variability (CV: {cv}%) suggests different user behaviors. Analyze: 1) Power users vs casual users, 2) Time-based segments, 3) Usage pattern clusters, 4) Personalization opportunities.'
                        }
                    }
                ]
            },
            
            // Financial Costs
            cost: {
                keywords: ['cost', 'expense', 'spend', 'budget', 'price', 'fee'],
                insights: [
                    {
                        condition: (stats) => stats.skewness > 1.0,
                        insight: {
                            type: 'action',
                            icon: '💸',
                            title: 'Cost Optimization Opportunity',
                            template: 'Right-skewed costs indicate few high-cost items drive total spend. Actions: 1) Analyze top {top_percentile}% cost drivers, 2) Negotiate volume discounts, 3) Alternative supplier evaluation, 4) Process efficiency improvements.'
                        }
                    },
                    {
                        condition: (stats) => stats.cv < 0.2,
                        insight: {
                            type: 'insight',
                            icon: '📋',
                            title: 'Standardized Costs',
                            template: 'Consistent cost structure (CV: {cv}%) suggests: 1) Standardized processes, 2) Predictable budgeting, 3) Efficient operations, 4) Potential for economies of scale.'
                        }
                    }
                ]
            },
            
            // Count Metrics
            count: {
                keywords: ['count', 'number', 'quantity', 'amount', 'total', 'sum'],
                insights: [
                    {
                        condition: (stats) => stats.mean < 1,
                        insight: {
                            type: 'warning',
                            icon: '📉',
                            title: 'Low Engagement Counts',
                            template: 'Average count below 1 ({mean}) suggests low engagement. Improvement strategies: 1) Simplify processes, 2) Increase incentives, 3) Remove barriers, 4) Improve value proposition.'
                        }
                    },
                    {
                        condition: (stats) => stats.skewness > 2.0,
                        insight: {
                            type: 'insight',
                            icon: '🎯',
                            title: 'Power User Pattern',
                            template: 'Heavily skewed counts indicate power users. Strategy: 1) Identify and nurture power users, 2) Gamification for broader engagement, 3) Create user tiers, 4) Learn from high-count behaviors.'
                        }
                    }
                ]
            }
        }
    },
    
    // ========================================
    // CATEGORICAL DATA INSIGHTS
    // ========================================
    categorical: {
        
        // Distribution Patterns
        distribution: {
            highlyConcentrated: {
                condition: (stats) => stats.topPercentage > 80,
                insight: {
                    type: 'insight',
                    icon: '🎯',
                    title: 'Highly Concentrated Distribution',
                    template: 'Dominant category "{topCategory}" represents {topPercentage}% of data. This indicates: 1) Clear market leader, 2) Strong customer preference, 3) Potential monopolistic position, 4) Focus optimization on this segment.'
                }
            },
            
            concentrated: {
                condition: (stats) => stats.topPercentage > 60 && stats.topPercentage <= 80,
                insight: {
                    type: 'insight',
                    icon: '📊',
                    title: 'Concentrated Distribution',
                    template: '"{topCategory}" leads with {topPercentage}% market share. Strategy: 1) Strengthen dominant position, 2) Defend against competitors, 3) Gradual expansion to other segments, 4) Leverage dominance for partnerships.'
                }
            },
            
            balanced: {
                condition: (stats) => stats.topPercentage >= 30 && stats.topPercentage <= 60 && stats.uniqueValues >= 3,
                insight: {
                    type: 'insight',
                    icon: '⚖️',
                    title: 'Balanced Distribution',
                    template: 'Well-balanced distribution across {uniqueValues} categories. This suggests: 1) Diverse market, 2) Multiple viable segments, 3) Competitive landscape, 4) Opportunities for targeted strategies.'
                }
            },
            
            fragmented: {
                condition: (stats) => stats.topPercentage < 30 && stats.uniqueValues > 5,
                insight: {
                    type: 'warning',
                    icon: '🧩',
                    title: 'Fragmented Distribution',
                    template: 'Highly fragmented across {uniqueValues} categories (top: {topPercentage}%). Consider: 1) Market consolidation opportunities, 2) Niche specialization, 3) Category grouping, 4) Portfolio rationalization.'
                }
            },
            
            binary: {
                condition: (stats) => stats.uniqueValues === 2,
                insight: {
                    type: 'action',
                    icon: '🔀',
                    title: 'Binary Classification Opportunity',
                    template: 'Perfect binary split: {binaryBreakdown}. Ideal for: 1) A/B testing, 2) Binary classification models, 3) Simple decision rules, 4) Clear segmentation strategies.'
                }
            }
        },
        
        // Business Context Patterns
        businessContext: {
            // Customer Segments
            segment: {
                keywords: ['segment', 'category', 'type', 'class', 'tier', 'level'],
                insights: [
                    {
                        condition: (stats) => stats.topPercentage > 70,
                        insight: {
                            type: 'action',
                            icon: '🎯',
                            title: 'Dominant Segment Strategy',
                            template: '"{topCategory}" segment dominates ({topPercentage}%). Strategy: 1) Deep specialization in this segment, 2) Premium offerings development, 3) Thought leadership positioning, 4) Expansion within segment before diversifying.'
                        }
                    },
                    {
                        condition: (stats) => stats.uniqueValues >= 4 && stats.topPercentage < 50,
                        insight: {
                            type: 'action',
                            icon: '🌈',
                            title: 'Multi-Segment Approach',
                            template: 'Diverse segments require tailored strategies. Actions: 1) Segment-specific value propositions, 2) Customized marketing messages, 3) Different pricing strategies, 4) Channel optimization per segment.'
                        }
                    }
                ]
            },
            
            // Geographic Data
            location: {
                keywords: ['location', 'region', 'city', 'state', 'country', 'area', 'zone', 'territory'],
                insights: [
                    {
                        condition: (stats) => stats.topPercentage > 60,
                        insight: {
                            type: 'action',
                            icon: '🌍',
                            title: 'Geographic Concentration',
                            template: 'Strong presence in "{topCategory}" ({topPercentage}%). Geographic strategy: 1) Dominate local market, 2) Study expansion to similar regions, 3) Local partnerships, 4) Regional customization of offerings.'
                        }
                    },
                    {
                        condition: (stats) => stats.uniqueValues > 10 && stats.topPercentage < 20,
                        insight: {
                            type: 'insight',
                            icon: '🗺️',
                            title: 'Wide Geographic Spread',
                            template: 'Presence across {uniqueValues} locations with broad distribution. Consider: 1) Regional clustering analysis, 2) Logistics optimization, 3) Local market strategies, 4) Regional management structure.'
                        }
                    }
                ]
            },
            
            // Product Categories
            product: {
                keywords: ['product', 'service', 'item', 'offering', 'solution'],
                insights: [
                    {
                        condition: (stats) => stats.topPercentage > 50,
                        insight: {
                            type: 'action',
                            icon: '📦',
                            title: 'Star Product Strategy',
                            template: '"{topCategory}" is your star product ({topPercentage}% of business). Actions: 1) Protect and enhance core offering, 2) Create product variants, 3) Bundle with other products, 4) Use as loss leader for portfolio growth.'
                        }
                    },
                    {
                        condition: (stats) => stats.uniqueValues > 20,
                        insight: {
                            type: 'warning',
                            icon: '📋',
                            title: 'Product Portfolio Complexity',
                            template: 'Large product portfolio ({uniqueValues} products) may create complexity. Consider: 1) Portfolio rationalization, 2) Product line consolidation, 3) Clear categorization, 4) Focus on top performers.'
                        }
                    }
                ]
            },
            
            // Status Fields
            status: {
                keywords: ['status', 'state', 'stage', 'phase', 'condition'],
                insights: [
                    {
                        condition: (stats, colName) => colName.toLowerCase().includes('churn') || colName.toLowerCase().includes('retention'),
                        insight: {
                            type: 'warning',
                            icon: '🔄',
                            title: 'Churn Analysis Critical',
                            template: 'Status distribution: {statusBreakdown}. Focus: 1) Identify churn predictors, 2) Early warning systems, 3) Retention campaigns for at-risk customers, 4) Win-back strategies for churned customers.'
                        }
                    },
                    {
                        condition: (stats) => stats.topPercentage > 80,
                        insight: {
                            type: 'insight',
                            icon: '✅',
                            title: 'Stable Status Distribution',
                            template: 'Most customers in "{topCategory}" status ({topPercentage}%). This suggests: 1) Stable operations, 2) Successful processes, 3) Monitor for status changes, 4) Maintain current service levels.'
                        }
                    }
                ]
            },
            
            // Channel/Source Data
            channel: {
                keywords: ['channel', 'source', 'medium', 'campaign', 'referrer'],
                insights: [
                    {
                        condition: (stats) => stats.topPercentage > 60,
                        insight: {
                            type: 'action',
                            icon: '📢',
                            title: 'Channel Concentration Risk',
                            template: 'Heavy reliance on "{topCategory}" channel ({topPercentage}%). Strategy: 1) Diversify acquisition channels, 2) Optimize dominant channel, 3) Test alternative channels, 4) Reduce single-channel dependency.'
                        }
                    },
                    {
                        condition: (stats) => stats.uniqueValues > 8 && stats.topPercentage < 30,
                        insight: {
                            type: 'action',
                            icon: '🎯',
                            title: 'Multi-Channel Optimization',
                            template: 'Diverse channel mix ({uniqueValues} channels). Optimize: 1) Measure channel ROI, 2) Attribution modeling, 3) Budget reallocation, 4) Channel-specific messaging, 5) Cross-channel synergies.'
                        }
                    }
                ]
            }
        },
        
        // Data Quality Patterns
        dataQuality: {
            highCardinality: {
                condition: (stats) => stats.cardinalityRatio > 0.8,
                insight: {
                    type: 'warning',
                    icon: '🔑',
                    title: 'High Cardinality - Likely Identifier',
                    template: 'Very high unique values ({uniqueValues}/{totalCount} = {cardinalityRatio}%). This appears to be an identifier field. Usage: 1) Primary key for joins, 2) Grouping operations, 3) Avoid in categorical analysis, 4) Consider for data relationships.'
                }
            },
            
            moderateCardinality: {
                condition: (stats) => stats.cardinalityRatio > 0.5 && stats.cardinalityRatio <= 0.8,
                insight: {
                    type: 'insight',
                    icon: '📊',
                    title: 'Moderate Cardinality',
                    template: 'Moderate cardinality ({cardinalityRatio}%) suggests semi-structured data. Consider: 1) Grouping similar categories, 2) Creating hierarchies, 3) Top-N analysis, 4) Rare category handling.'
                }
            }
        }
    },
    
    // ========================================
    // CORRELATION INSIGHTS
    // ========================================
    correlation: {
        strong: {
            condition: (corr) => Math.abs(corr) > 0.8,
            insight: {
                type: 'insight',
                icon: '🔗',
                title: 'Strong Correlation',
                template: 'Very strong {direction} correlation ({correlation}%). This suggests: 1) Potential causal relationship, 2) Redundant variables, 3) Predictive modeling opportunity, 4) Business process connection.'
            }
        },
        
        moderate: {
            condition: (corr) => Math.abs(corr) > 0.5 && Math.abs(corr) <= 0.8,
            insight: {
                type: 'insight',
                icon: '🔗',
                title: 'Moderate Correlation',
                template: 'Moderate {direction} correlation ({correlation}%). Consider: 1) Factor in decision making, 2) Monitor relationship stability, 3) Investigate underlying drivers, 4) Potential for predictive features.'
            }
        },
        
        weak: {
            condition: (corr) => Math.abs(corr) > 0.3 && Math.abs(corr) <= 0.5,
            insight: {
                type: 'insight',
                icon: '🔗',
                title: 'Weak Correlation',
                template: 'Weak {direction} correlation ({correlation}%). This may indicate: 1) Subtle relationship, 2) Non-linear association, 3) Contextual dependency, 4) Worth deeper investigation.'
            }
        }
    }
};

// Helper functions for insight generation
const INSIGHT_HELPERS = {
    calculateThresholds: (stats) => ({
        high_threshold: stats.mean + stats.std,
        med_threshold: stats.mean,
        low_threshold: stats.mean - stats.std,
        improvement_threshold: stats.mean - (stats.std * 0.5),
        top_percentile: 10
    }),
    
    formatTemplateString: (template, data) => {
        return template.replace(/{(\w+)}/g, (match, key) => {
            if (data[key] !== undefined) {
                if (typeof data[key] === 'number') {
                    return typeof data[key] === 'number' && data[key] % 1 !== 0 ? 
                           data[key].toFixed(2) : data[key].toString();
                }
                return data[key];
            }
            return match;
        });
    },
    
    getCorrelationDirection: (correlation) => {
        return correlation > 0 ? 'positive' : 'negative';
    },
    
    formatPercentage: (value) => {
        return (value * 100).toFixed(1) + '%';
    }
};

// Export for use in main application
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { INSIGHTS_CONFIG, INSIGHT_HELPERS };
}