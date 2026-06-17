use crate::error::Result;
use async_trait::async_trait;
use indexmap::IndexMap;
use ndarray::ArrayD;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorDtype {
    F32,
    F64,
    I32,
    I64,
    Bool,
    String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimSize {
    Fixed(usize),
    Dynamic,
    Named(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: TensorDtype,
    pub shape: Vec<DimSize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorValue {
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    I32(ArrayD<i32>),
    I64(ArrayD<i64>),
    Bool(ArrayD<bool>),
    String(ArrayD<String>),
}

impl TensorValue {
    pub fn dtype(&self) -> TensorDtype {
        match self {
            Self::F32(_) => TensorDtype::F32,
            Self::F64(_) => TensorDtype::F64,
            Self::I32(_) => TensorDtype::I32,
            Self::I64(_) => TensorDtype::I64,
            Self::Bool(_) => TensorDtype::Bool,
            Self::String(_) => TensorDtype::String,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(v) => v.shape(),
            Self::F64(v) => v.shape(),
            Self::I32(v) => v.shape(),
            Self::I64(v) => v.shape(),
            Self::Bool(v) => v.shape(),
            Self::String(v) => v.shape(),
        }
    }

    pub fn ndim(&self) -> usize {
        self.shape().len()
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TensorBatch {
    pub tensors: IndexMap<String, TensorValue>,
}

impl TensorBatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, value: TensorValue) -> Option<TensorValue> {
        self.tensors.insert(name.into(), value)
    }

    pub fn get(&self, name: &str) -> Option<&TensorValue> {
        self.tensors.get(name)
    }

    pub fn into_tensors(self) -> IndexMap<String, TensorValue> {
        self.tensors
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

#[async_trait]
pub trait RawTensorModel: Send + Sync {
    async fn run(&self, inputs: &TensorBatch) -> Result<TensorBatch>;

    async fn run_batch(&self, inputs: &[TensorBatch]) -> Result<Vec<TensorBatch>> {
        let mut out = Vec::with_capacity(inputs.len());
        for input in inputs {
            out.push(self.run(input).await?);
        }
        Ok(out)
    }

    fn max_batch_size(&self) -> usize {
        1
    }

    fn input_signature(&self) -> &[TensorSpec];
    fn output_signature(&self) -> &[TensorSpec];

    async fn warmup(&self) -> Result<()> {
        Ok(())
    }

    /// Names of the ONNX Runtime execution providers **requested** when this
    /// session was loaded, in priority order — e.g.
    /// `["cuda", "cpu"]`, `["coreml", "cpu"]`, `["cpu"]`.
    ///
    /// # What this does and doesn't tell you
    ///
    /// - **Tells you**: which EPs the session was *configured to try*. A
    ///   session whose request list is `["cpu"]` will not engage CUDA no
    ///   matter what hardware is present — useful for catching "the catalog
    ///   spec didn't ask for the GPU" misconfigurations.
    /// - **Does NOT tell you**: which EP actually attached. If the request
    ///   list is `["cuda", "cpu"]` and CUDA failed silently because the
    ///   loaded ORT tarball lacks `libonnxruntime_providers_shared.so`,
    ///   this still reports `["cuda", "cpu"]`. The ORT 2.0 Rust binding
    ///   doesn't yet expose `Session::providers()`.
    ///
    /// To verify *actual* attachment, watch ORT's tracing logs for
    /// `Successfully created <Provider>` messages, or run a model that
    /// crashes-not-falls-back when the EP isn't real.
    ///
    /// Default returns an empty `Vec` for runners that don't track this
    /// (mocks, future non-ORT backends).
    fn active_execution_providers(&self) -> Vec<String> {
        Vec::new()
    }
}
