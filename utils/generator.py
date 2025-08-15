import numpy as np
import random
import string
import pandas as pd
import math
from datetime import timedelta
from scipy.stats import truncnorm
from faker import Faker

# optional regex-based generator
try:
    import rstr  # provides rstr.xeger for regex -> string
    HAS_RSTR = True
except Exception:
    HAS_RSTR = False

_fake = Faker()

# -----------------------------
# Helpers
# -----------------------------

def _seed_all(seed: int | None):
    """Set seeds for all random number generators for reproducibility."""
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    _fake.seed_instance(seed)

def _get_numeric_defaults(field, is_int=False):
    """Get default values for numeric fields with improved statistical handling."""
    min_val = field.get("min", 0 if is_int else 0.0)
    max_val = field.get("max", 100 if is_int else 100.0)
    
    # Ensure min <= max
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    
    range_ = max(max_val - min_val, 1e-6)  # avoid zero division
    
    # Use provided mean/std or calculate defaults
    if "mean" in field and "std" in field:
        mean = float(field["mean"])
        std = float(field["std"])
        
        # Ensure std is reasonable for the range
        max_std = range_ / 3  # 3-sigma rule
        if std > max_std:
            print(f"Warning: Standard deviation {std} is too large for range [{min_val}, {max_val}], capping at {max_std}")
            std = max_std
    else:
        mean = field.get("mean", (min_val + max_val) / 2)
        std = field.get("std", max(range_ / 6.0, 1e-6))  # Default: range/6 (fits ~99.7% within bounds)
    
    # Ensure mean is within bounds
    mean = max(min_val, min(max_val, mean))
    
    return float(min_val), float(max_val), float(mean), float(std)

def _truncnorm(a, b, loc, scale, size):
    """Generate from truncated normal distribution with correct parameter order."""
    return truncnorm.rvs(a, b, loc=loc, scale=scale, size=size)

# -----------------------------
# Data Quality Analysis Functions
# -----------------------------

def analyze_data_quality(df):
    """Analyze data quality metrics for all fields."""
    quality_stats = {}
    
    for col in df.columns:
        col_data = df[col]
        total_count = len(col_data)
        
        # Missing values
        missing_count = col_data.isna().sum() if hasattr(col_data, 'isna') else sum(1 for x in col_data if x is None)
        missing_pct = (missing_count / total_count * 100) if total_count > 0 else 0
        
        # Duplicates
        if hasattr(col_data, 'duplicated'):
            duplicate_count = col_data.duplicated().sum()
        else:
            # For lists with None values
            non_null_values = [x for x in col_data if x is not None]
            duplicate_count = len(non_null_values) - len(set(non_null_values)) if non_null_values else 0
        
        duplicate_pct = (duplicate_count / total_count * 100) if total_count > 0 else 0
        
        stats = {
            'total_count': total_count,
            'missing_count': missing_count,
            'missing_pct': missing_pct,
            'duplicate_count': duplicate_count,
            'duplicate_pct': duplicate_pct,
            'unique_count': len(set(x for x in col_data if x is not None)),
            'data_type': str(type(col_data.iloc[0] if hasattr(col_data, 'iloc') and len(col_data) > 0 else col_data[0] if col_data else None))
        }
        
        # Outlier detection for numeric fields (FIXED - exclude boolean)
        if pd.api.types.is_numeric_dtype(col_data) and col_data.dtype != 'bool':
            non_null_numeric = col_data.dropna() if hasattr(col_data, 'dropna') else [x for x in col_data if x is not None and isinstance(x, (int, float))]
            
            if len(non_null_numeric) > 0:
                non_null_numeric = pd.Series(non_null_numeric) if not isinstance(non_null_numeric, pd.Series) else non_null_numeric
                
                # IQR method for outlier detection
                Q1 = non_null_numeric.quantile(0.25)
                Q3 = non_null_numeric.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = non_null_numeric[(non_null_numeric < lower_bound) | (non_null_numeric > upper_bound)]
                outlier_count = len(outliers)
                outlier_pct = (outlier_count / len(non_null_numeric) * 100) if len(non_null_numeric) > 0 else 0
                
                stats.update({
                    'outlier_count': outlier_count,
                    'outlier_pct': outlier_pct,
                    'mean': non_null_numeric.mean(),
                    'std': non_null_numeric.std(),
                    'min': non_null_numeric.min(),
                    'max': non_null_numeric.max()
                })
        
        quality_stats[col] = stats
    
    return quality_stats

# -----------------------------
# Data Imperfection Functions
# -----------------------------

def apply_missing_data(values, missing_pct=5.0, missing_pattern="random"):
    """
    Apply missing data patterns to any field type.
    
    Args:
        values: Original field values (list or numpy array)
        missing_pct: Percentage of values to make missing (0-100)
        missing_pattern: 'random', 'block', or 'tail'
    
    Returns:
        Modified values with missing data (None for missing values)
    """
    if missing_pct <= 0:
        return values
    
    values = list(values)  # Convert to list for easier manipulation
    n = len(values)
    n_missing = max(1, int(n * missing_pct / 100))
    
    if missing_pattern == "random":
        # Random missing values throughout the dataset
        missing_indices = random.sample(range(n), min(n_missing, n))
    elif missing_pattern == "block":
        # Contiguous block of missing values
        start_idx = random.randint(0, max(0, n - n_missing))
        missing_indices = list(range(start_idx, min(start_idx + n_missing, n)))
    elif missing_pattern == "tail":
        # Missing values at the end (simulates data collection issues)
        missing_indices = list(range(max(0, n - n_missing), n))
    else:
        missing_indices = []
    
    # Apply missing values
    for idx in missing_indices:
        values[idx] = None
    
    print(f"Applied {len(missing_indices)} missing values ({len(missing_indices)/n*100:.1f}%) with pattern '{missing_pattern}'")
    return values

def apply_outliers_numeric(values, outlier_pct=2.0, outlier_method="extreme"):
    """
    Apply outliers to numeric fields only.
    
    Args:
        values: Numeric field values (numpy array or list)
        outlier_pct: Percentage of values to convert to outliers (0-100)
        outlier_method: 'extreme', 'mild', or 'random'
    
    Returns:
        Modified values with outliers
    """
    if outlier_pct <= 0:
        return values
    
    values = np.array(values, dtype=float)
    n = len(values)
    n_outliers = max(1, int(n * outlier_pct / 100))
    
    # Calculate statistics for outlier generation
    mean_val = np.mean(values)
    std_val = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    range_val = max_val - min_val
    
    # Select random indices for outliers
    outlier_indices = random.sample(range(n), min(n_outliers, n))
    
    for idx in outlier_indices:
        if outlier_method == "extreme":
            # 3+ standard deviations away from mean
            direction = random.choice([-1, 1])
            outlier_val = mean_val + direction * (3 + random.uniform(0, 2)) * std_val
        elif outlier_method == "mild":
            # 2-3 standard deviations away from mean
            direction = random.choice([-1, 1])
            outlier_val = mean_val + direction * random.uniform(2, 3) * std_val
        else:  # random
            # Random extreme values outside normal range
            if random.random() < 0.5:
                outlier_val = min_val - random.uniform(0.5, 2.0) * range_val
            else:
                outlier_val = max_val + random.uniform(0.5, 2.0) * range_val
        
        values[idx] = outlier_val
    
    print(f"Applied {len(outlier_indices)} outliers ({len(outlier_indices)/n*100:.1f}%) using method '{outlier_method}'")
    return values

def apply_duplicates(values, duplicate_pct=5.0, duplicate_strategy="common_values"):
    """
    Apply duplicate values to numeric and string fields.
    
    Args:
        values: Field values (list or numpy array)
        duplicate_pct: Percentage of values to convert to duplicates (0-100)
        duplicate_strategy: 'common_values', 'random_pick', or 'single_value'
    
    Returns:
        Modified values with increased duplicates
    """
    if duplicate_pct <= 0:
        return values
    
    values = list(values)  # Convert to list
    n = len(values)
    n_duplicates = max(1, int(n * duplicate_pct / 100))
    
    # Select indices to convert to duplicates
    duplicate_indices = random.sample(range(n), min(n_duplicates, n))
    
    if duplicate_strategy == "common_values":
        # Use most common existing values
        from collections import Counter
        value_counts = Counter(v for v in values if v is not None)
        if value_counts:
            common_values = [val for val, _ in value_counts.most_common(min(5, len(value_counts)))]
            for idx in duplicate_indices:
                values[idx] = random.choice(common_values)
        
    elif duplicate_strategy == "random_pick":
        # Pick random existing values to duplicate
        non_null_values = [v for v in values if v is not None]
        if non_null_values:
            for idx in duplicate_indices:
                values[idx] = random.choice(non_null_values)
    
    else:  # single_value
        # Create clusters of the same value
        if values:
            target_value = random.choice([v for v in values if v is not None]) if any(v is not None for v in values) else values[0]
            for idx in duplicate_indices:
                values[idx] = target_value
    
    print(f"Applied duplicates to {len(duplicate_indices)} values ({len(duplicate_indices)/n*100:.1f}%) using strategy '{duplicate_strategy}'")
    return values

def apply_imperfections(field_config, values):
    """
    Apply all configured imperfections to a field's values.
    
    Args:
        field_config: Field configuration dictionary with imperfection settings
        values: Generated field values
    
    Returns:
        Values with applied imperfections
    """
    field_type = field_config.get("type", "").lower()
    
    # Apply outliers (numeric fields only)
    if field_type in ["integer", "float"] and field_config.get("outlier_pct", 0) > 0:
        outlier_method = field_config.get("outlier_method", "extreme")
        values = apply_outliers_numeric(values, field_config["outlier_pct"], outlier_method)
    
    # Apply duplicates (numeric and string fields only)
    if field_type in ["integer", "float", "string"] and field_config.get("duplicate_pct", 0) > 0:
        duplicate_strategy = field_config.get("duplicate_strategy", "common_values")
        values = apply_duplicates(values, field_config["duplicate_pct"], duplicate_strategy)
    
    # Apply missing data (all field types)
    if field_config.get("missing_pct", 0) > 0:
        missing_pattern = field_config.get("missing_pattern", "random")
        values = apply_missing_data(values, field_config["missing_pct"], missing_pattern)
    
    return values

# -----------------------------
# Original Generators (unchanged)
# -----------------------------

def generate_integer(field, size):
    """Generate integer values with optional statistical constraints."""
    min_val, max_val, mean, std = _get_numeric_defaults(field, is_int=True)
    
    # Calculate standardized bounds (z-scores)
    a = (min_val - mean) / std
    b = (max_val - mean) / std
    
    # Generate with correct parameter order
    data = _truncnorm(a, b, mean, std, size)
    return np.round(np.clip(data, min_val, max_val)).astype(int)

def generate_float(field, size):
    """Generate float values with optional statistical constraints."""
    min_val, max_val, mean, std = _get_numeric_defaults(field, is_int=False)
    
    # Calculate standardized bounds (z-scores)  
    a = (min_val - mean) / std
    b = (max_val - mean) / std
    
    # Generate with correct parameter order
    data = _truncnorm(a, b, mean, std, size)
    
    # Clip to bounds (extra safety)
    data = np.clip(data, min_val, max_val)
    
    # Round if decimal_places specified
    if "decimal_places" in field:
        data = np.round(data, int(field["decimal_places"]))
    
    return data

def generate_categorical(field, size):
    """Generate categorical data with optional probability weights."""
    cats = field.get("categories", ["A", "B", "C"])
    probs_raw = field.get("probs")
    probs = None
    
    if probs_raw:
        try:
            probs = [float(p) for p in probs_raw]
            if len(probs) == len(cats) and sum(probs) > 0:
                # Normalize probabilities
                probs = [p / sum(probs) for p in probs]
            else:
                probs = None
        except (ValueError, TypeError):
            probs = None
    
    return np.random.choice(cats, size=size, p=probs)

def generate_string(field, size):
    """Generate string data using templates, regex patterns, or random strings."""
    # 1) Template (highest priority)
    template = field.get("template")
    if template:
        template_map = {
            "name": lambda: _fake.name(),
            "email": lambda: _fake.email(),
            "phone": lambda: _fake.phone_number(),
            "address": lambda: _fake.address().replace("\n", ", "),
            "company": lambda: _fake.company(),
            "job": lambda: _fake.job(),
            "sentence": lambda: _fake.sentence(),
            "paragraph": lambda: _fake.paragraph()
        }
        
        if template in template_map:
            return [template_map[template]() for _ in range(size)]
    
    # 2) Regex pattern (if available and rstr installed)
    pattern = field.get("pattern")
    if pattern and HAS_RSTR:
        try:
            return [rstr.xeger(pattern) for _ in range(size)]
        except Exception:
            pass  # fall through to random strings
    
    # 3) Fallback random alphanumerics
    length = int(field.get("length", 8))
    alphabet = string.ascii_letters + string.digits
    return [
        "".join(random.choices(alphabet, k=length))
        for _ in range(size)
    ]

def generate_datetime(field, size):
    """Generate datetime data in various formats."""
    mode = field.get("datetime_mode", "date")
    
    if mode in ("date", "datetime"):
        start = pd.to_datetime(field.get("start_date", "2020-01-01"))
        end = pd.to_datetime(field.get("end_date", "2024-01-01"))
        
        if end < start:
            start, end = end, start
        
        days = max((end - start).days, 1)
        
        if mode == "date":
            vals = [start + timedelta(days=random.randint(0, days)) for _ in range(size)]
            return [v.date().isoformat() for v in vals]
        else:  # datetime
            seconds_range = max(int((end - start).total_seconds()), 1)
            vals = [start + pd.to_timedelta(random.randint(0, seconds_range), unit="s") for _ in range(size)]
            return [v.strftime("%Y-%m-%d %H:%M:%S") for v in vals]
    
    # mode == "time"
    def _rand_time():
        h, m, s = random.randint(0, 23), random.randint(0, 59), random.randint(0, 59)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    return [_rand_time() for _ in range(size)]

def generate_boolean(field, size):
    """Generate boolean data with optional probability bias."""
    p = float(field.get("true_probability", 0.5))
    p = max(0.0, min(1.0, p))  # clamp to [0,1]
    return [random.random() < p for _ in range(size)]

# -----------------------------
# Updated Dispatcher
# -----------------------------

def generate_field(field, size):
    """Route field generation to appropriate type-specific function with imperfections."""
    dtype = field["type"].lower()
    
    generators = {
        "integer": generate_integer,
        "float": generate_float,
        "categorical": generate_categorical,
        "string": generate_string,
        "date/time": generate_datetime,
        "date": lambda f, s: generate_datetime({**f, "datetime_mode": "date"}, s),
        "datetime": lambda f, s: generate_datetime({**f, "datetime_mode": "datetime"}, s),
        "time": lambda f, s: generate_datetime({**f, "datetime_mode": "time"}, s),
        "boolean": generate_boolean
    }
    
    if dtype not in generators:
        raise ValueError(f"Unsupported data type: {dtype}")
    
    # Generate clean data first
    values = generators[dtype](field, size)
    
    # Apply imperfections if configured
    # values = apply_imperfections(field, values)
    
    return values

def _is_numeric_field(field_config):
    """Check if a field configuration represents a numeric type."""
    return field_config and field_config.get("type", "").lower() in ["integer", "float"]

def _is_numeric_array(arr):
    """Safely check if an array contains numeric data."""
    if not isinstance(arr, np.ndarray):
        return False
    return np.issubdtype(arr.dtype, np.number)

# -----------------------------
# Categorical Relationships Functions
# -----------------------------

def apply_categorical_relationships(data, categorical_relationships, fields):
    """
    Apply categorical relationships to modify numeric fields based on categorical field values.
    
    Args:
        data: Dictionary containing generated field data
        categorical_relationships: List of categorical relationship configurations
        fields: List of field configurations for type checking
        
    Returns:
        Modified data dictionary
    """
    for rel in categorical_relationships:
        cat_field = rel["categorical_field"]
        target_field = rel["target_field"]
        mappings = rel["category_mappings"]
        
        if cat_field not in data or target_field not in data:
            print(f"Warning: Fields '{cat_field}' or '{target_field}' not found for categorical relationship")
            continue
        
        print(f"Applying categorical relationship: {cat_field} -> {target_field}")
        
        # Get categorical values
        cat_values = data[cat_field]
        target_values = np.array([v if v is not None else 0.0 for v in data[target_field]], dtype=float)
        
        # Apply mappings for each category
        for i, cat_val in enumerate(cat_values):
            if cat_val is None:  # Skip missing categorical values
                continue
                
            if cat_val in mappings:
                mapping = mappings[cat_val]
                
                # Extract parameters
                if "range" in mapping:
                    min_val, max_val = mapping["range"]
                    target_values[i] = random.uniform(min_val, max_val)
                elif "mean" in mapping:
                    mean = mapping["mean"]
                    std = mapping.get("std", mean * 0.2)  # Default std is 20% of mean
                    target_values[i] = max(0, random.normalvariate(mean, std))
        
        # Update target field, preserving original type
        target_field_cfg = next((f for f in fields if f["name"] == target_field), None)
        if target_field_cfg and target_field_cfg["type"].lower() == "integer":
            target_values = np.round(target_values).astype(int)
        
        data[target_field] = target_values
        
    return data

# -----------------------------
# Updated Main Generation Function
# -----------------------------

def generate_data(fields, size, seed=None, relationships=None, categorical_relationships=None):
    """
    Generate synthetic dataset with optional field relationships and imperfections.
    
    Args:
        fields: List of field configuration dictionaries
        size: Number of records to generate
        seed: Random seed for reproducibility
        relationships: List of relationship configurations for dependent fields
        categorical_relationships: List of categorical relationship configurations
    
    Returns:
        pandas.DataFrame: Generated synthetic data with imperfections
    """
    _seed_all(seed)
    data = {}
    relationships = relationships or []
    categorical_relationships = categorical_relationships or []
    
    # Generate ALL fields first (including imperfections)
    for field in fields:
        if field.get("name"):
            print(f"Generating field: {field['name']}")  # Debug log
            data[field["name"]] = generate_field(field, size)
    
    print(f"Generated independent fields: {list(data.keys())}")  # Debug log
    
    # Apply mathematical relationships as transformations on top of existing data
    for rel in relationships:
        target = rel["target"]
        source_name = rel["source"]
        func_type = rel["function"]
        param = rel.get("param", 1)
        noise_pct = rel.get("noise", 5)
        
        print(f"Applying relationship: {source_name} -> {target} ({func_type})")  # Debug log
        
        if source_name not in data:
            print(f"Warning: Source field '{source_name}' not found for relationship '{target}'.")
            continue
        
        # Convert to float array safely, handling None values from missing data
        try:
            source_values = data[source_name]
            if isinstance(source_values, list):
                # Handle None values from missing data
                clean_values = [v if v is not None else 0.0 for v in source_values]
                src_vals = np.array(clean_values, dtype=float)
            else:
                src_vals = np.array(source_values, dtype=float)
        except (ValueError, TypeError) as e:
            print(f"Error: Cannot convert source field '{source_name}' to numeric: {e}")
            continue
            
        print(f"Source values sample: {src_vals[:3]}")  # Debug log
        
        # Apply mathematical relationship with improved error handling
        try:
            if func_type == "Direct Proportion":
                tgt_vals = src_vals * param
                
            elif func_type == "Inverse Proportion":
                # Handle division by zero
                tgt_vals = np.divide(
                    param, src_vals,
                    out=np.zeros_like(src_vals),
                    where=src_vals != 0
                )
                
            elif func_type == "Logarithmic":
                base = param if param > 0 else math.e
                # Handle non-positive values
                tgt_vals = np.where(
                    src_vals > 0,
                    np.log(src_vals) / np.log(base),
                    0
                )
                
            elif func_type == "Exponential (source^k)":
                # Clip source values to prevent overflow
                safe_src = np.clip(src_vals, -50, 50)
                tgt_vals = np.power(safe_src, param)
                
            elif func_type == "Exponential (k^source)":
                # Ensure positive base and handle overflow
                base = abs(param) if param != 0 else 2.0
                print(f"Using base: {base} for exponential relationship")  # Debug log
                
                # More conservative clipping to prevent overflow
                safe_src = np.clip(src_vals, -10, 10)
                
                try:
                    tgt_vals = np.power(base, safe_src)
                    
                    # Handle any potential overflow/underflow
                    tgt_vals = np.where(np.isfinite(tgt_vals), tgt_vals, 1.0)
                    
                    # Ensure all exponential results are positive
                    tgt_vals = np.abs(tgt_vals)
                    
                except (OverflowError, ValueError) as e:
                    print(f"Overflow in exponential calculation, using fallback values: {e}")
                    tgt_vals = np.ones_like(safe_src)
                    
            else:
                raise ValueError(f"Unknown relationship type: {func_type}")
            
            # Validate results before adding noise
            if np.any(np.isnan(tgt_vals)) or np.any(np.isinf(tgt_vals)):
                print(f"Warning: Invalid values in {target}, replacing with fallback")
                tgt_vals = np.ones_like(src_vals)
            
            # Add noise for realistic approximation
            if noise_pct > 0 and len(tgt_vals) > 0:
                mean_abs = np.mean(np.abs(tgt_vals))
                if mean_abs > 0:  # Avoid noise when all values are zero
                    noise = np.random.normal(0, (noise_pct / 100) * mean_abs, size=len(tgt_vals))
                    tgt_vals += noise
            
            # Final validation and type matching
            # Ensure no negative values for exponential relationships
            if func_type == "Exponential (k^source)":
                tgt_vals = np.abs(tgt_vals)
            
            # Match target field data type
            target_field_cfg = next((f for f in fields if f["name"] == target), None)
            if target_field_cfg and target_field_cfg["type"].lower() == "integer":
                tgt_vals = np.round(tgt_vals).astype(int)
            
            print(f"Target values sample: {tgt_vals[:3]}")  # Debug log
            
            # Apply imperfections to relationship target if configured
            if target_field_cfg:
                tgt_vals = apply_imperfections(target_field_cfg, tgt_vals)
            
            data[target] = tgt_vals
            
        except Exception as e:
            print(f"Error applying relationship {func_type} from {source_name} to {target}: {str(e)}")
            # Keep original generated values instead of failing
            continue
    
    # Apply categorical relationships
    if categorical_relationships:
        data = apply_categorical_relationships(data, categorical_relationships, fields)
    
    # Apply imperfections to all fields after relationships
    # This ensures that imperfections are applied after all relationships are established
    for field in fields:
        if field.get("name") and field["name"] in data:
            data[field["name"]] = apply_imperfections(field, data[field["name"]])

    
    # Safe data validation with proper type checking
    for name, values in data.items():
        if isinstance(values, np.ndarray):
            # Only check for NaN/inf on numeric arrays
            if _is_numeric_array(values):
                if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                    print(f"Warning: Field {name} contains invalid numeric values after processing")
                    # Regenerate the field if it has invalid values
                    field_config = next((f for f in fields if f["name"] == name), None)
                    if field_config:
                        data[name] = generate_field(field_config, size)
        elif isinstance(values, list):
            # For list data, this is normal (None values are valid for missing data)
            pass
    
    return pd.DataFrame(data)

# -----------------------------
# Validation Functions
# -----------------------------

def validate_fields(fields):
    """Validate field configurations and return any issues found."""
    issues = []
    field_names = set()
    
    for i, field in enumerate(fields):
        if not field.get("name"):
            issues.append(f"Field {i+1}: Name is required")
            continue
        
        if field["name"] in field_names:
            issues.append(f"Field {i+1}: Duplicate field name '{field['name']}'")
        field_names.add(field["name"])
        
        if not field.get("type"):
            issues.append(f"Field '{field['name']}': Type is required")
    
    return issues

def validate_relationships(relationships, fields):
    """Validate relationship configurations."""
    issues = []
    field_names = {f["name"] for f in fields if f.get("name")}
    numeric_types = {"integer", "float"}
    
    for i, rel in enumerate(relationships):
        rel_name = f"Relationship {i+1}"
        
        # Check required fields
        if not rel.get("target") or not rel.get("source"):
            issues.append(f"{rel_name}: Target and source fields are required")
            continue
        
        # Check field existence
        if rel["target"] not in field_names:
            issues.append(f"{rel_name}: Target field '{rel['target']}' does not exist")
        if rel["source"] not in field_names:
            issues.append(f"{rel_name}: Source field '{rel['source']}' does not exist")
        
        # Check that fields are numeric
        target_field = next((f for f in fields if f["name"] == rel["target"]), None)
        source_field = next((f for f in fields if f["name"] == rel["source"]), None)
        
        if target_field and target_field["type"].lower() not in numeric_types:
            issues.append(f"{rel_name}: Target field must be numeric (integer or float)")
        if source_field and source_field["type"].lower() not in numeric_types:
            issues.append(f"{rel_name}: Source field must be numeric (integer or float)")
    
    return issues

def validate_categorical_relationships(categorical_relationships, fields):
    """Validate categorical relationship configurations."""
    issues = []
    field_names = {f["name"] for f in fields if f.get("name")}
    
    for i, rel in enumerate(categorical_relationships):
        rel_name = f"Categorical Relationship {i+1}"
        
        # Check required fields
        if not rel.get("categorical_field") or not rel.get("target_field"):
            issues.append(f"{rel_name}: Categorical and target fields are required")
            continue
        
        # Check field existence
        if rel["categorical_field"] not in field_names:
            issues.append(f"{rel_name}: Categorical field '{rel['categorical_field']}' does not exist")
        if rel["target_field"] not in field_names:
            issues.append(f"{rel_name}: Target field '{rel['target_field']}' does not exist")
        
        # Check field types
        cat_field = next((f for f in fields if f["name"] == rel["categorical_field"]), None)
        target_field = next((f for f in fields if f["name"] == rel["target_field"]), None)
        
        if cat_field and cat_field["type"].lower() != "categorical":
            issues.append(f"{rel_name}: Source field must be categorical")
        if target_field and target_field["type"].lower() not in ["integer", "float"]:
            issues.append(f"{rel_name}: Target field must be numeric (integer or float)")
        
        # Check category mappings
        if not rel.get("category_mappings"):
            issues.append(f"{rel_name}: Category mappings are required")
    
    return issues