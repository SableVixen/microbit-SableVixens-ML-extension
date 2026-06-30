//% color=#696969 icon="\uf2db" block="SableVixen's ML extension"
namespace ml {

    function computeChecksum(s: string): number {
        let sum = 0;
        for (let i = 0; i < s.length; i++) {
            sum = (sum + s.charCodeAt(i)) & 0xff;
        }
        return sum;
    }

    function compressNumber(n: number): string {
        return "" + Math.round(n * 100) / 100;
    }

    export class Dataset {
        labels: string[];
        samples: number[][];

        constructor() {
            this.labels = [];
            this.samples = [];
        }

        add(label: string, features: number[]) {
            this.labels.push(label);
            this.samples.push(features);
        }
    }

    //% block="Create dataset"
    export function createDataset(): Dataset {
        return new Dataset();
    }

    //% block="Add sample to %ds label %label with features %f1 %f2 %f3"
    export function addSample3(ds: Dataset, label: string, f1: number, f2: number, f3: number) {
        ds.add(label, [f1, f2, f3]);
    }

    //% block="Extract accelerometer features"
    export function accelFeatures(): number[] {
        let ax = input.acceleration(Dimension.X);
        let ay = input.acceleration(Dimension.Y);
        let az = input.acceleration(Dimension.Z);
        return [ax, ay, az];
    }

    //% block="Extract sound energy"
    export function soundEnergy(): number {
        return input.soundLevel();
    }

    export class KNN {
        ds: Dataset;
        k: number;

        constructor(ds: Dataset, k: number) {
            this.ds = ds;
            this.k = k;
        }

        classify(features: number[]): string {
            let distances: number[] = [];
            let labels: string[] = [];

            for (let i = 0; i < this.ds.samples.length; i++) {
                let s = this.ds.samples[i];
                let sum = 0;
                for (let j = 0; j < s.length; j++) {
                    let diff = s[j] - features[j];
                    sum += diff * diff;
                }
                distances.push(Math.sqrt(sum));
                labels.push(this.ds.labels[i]);
            }

            let voteCount: { [key: string]: number } = {};

            for (let n = 0; n < this.k; n++) {
                let bestIndex = 0;
                let bestDist = 999999;

                for (let i = 0; i < distances.length; i++) {
                    if (distances[i] < bestDist) {
                        bestDist = distances[i];
                        bestIndex = i;
                    }
                }

                let lbl = labels[bestIndex];
                voteCount[lbl] = (voteCount[lbl] || 0) + 1;

                distances[bestIndex] = 999999;
            }

            let keys = Object.keys(voteCount);
            let bestLabel = "";
            let bestVotes = 0;

            for (let i = 0; i < keys.length; i++) {
                let lbl = keys[i];
                if (voteCount[lbl] > bestVotes) {
                    bestVotes = voteCount[lbl];
                    bestLabel = lbl;
                }
            }

            return bestLabel;
        }

        savePayload(): string {
            let parts: string[] = [];
            parts.push("" + this.k);
            parts.push("" + this.ds.samples.length);
            for (let i = 0; i < this.ds.samples.length; i++) {
                let s = this.ds.samples[i];
                let lbl = this.ds.labels[i];
                let line = lbl;
                for (let j = 0; j < s.length; j++) {
                    line = line + "," + s[j];
                }
                parts.push(line);
            }
            return parts.join("|");
        }

        static loadPayload(payload: string): KNN {
            let parts = payload.split("|");
            if (parts.length < 2) return new KNN(new Dataset(), 1);
            let k = parseInt(parts[0]);
            let count = parseInt(parts[1]);
            let ds = new Dataset();
            for (let i = 0; i < count; i++) {
                let line = parts[2 + i];
                let seg = line.split(",");
                let lbl = seg[0];
                let feats: number[] = [];
                for (let j = 1; j < seg.length; j++) {
                    feats.push(parseInt(seg[j]));
                }
                ds.add(lbl, feats);
            }
            return new KNN(ds, k);
        }
    }

    //% block="Create KNN classifier from %ds with k %k"
    export function createKNN(ds: Dataset, k: number): KNN {
        return new KNN(ds, k);
    }

    //% block="KNN %knn classify features %f1 %f2 %f3"
    export function knnClassify3(knn: KNN, f1: number, f2: number, f3: number): string {
        return knn.classify([f1, f2, f3]);
    }

    export class Perceptron {
        weights: number[];
        bias: number;

        constructor(size: number) {
            this.weights = [];
            for (let i = 0; i < size; i++) {
                this.weights.push(Math.randomRange(-10, 10) / 10);
            }
            this.bias = 0;
        }

        train(features: number[], label: number, lr: number) {
            let pred = this.predict(features);
            let error = label - pred;

            for (let i = 0; i < this.weights.length; i++) {
                this.weights[i] += lr * error * features[i];
            }
            this.bias += lr * error;
        }

        predict(features: number[]): number {
            let sum = this.bias;
            for (let i = 0; i < this.weights.length; i++) {
                sum += this.weights[i] * features[i];
            }
            return sum > 0 ? 1 : 0;
        }
    }

    //% block="Create perceptron with %n inputs"
    export function createPerceptron(n: number): Perceptron {
        return new Perceptron(n);
    }

    //% block="Train perceptron %p with features %f1 %f2 %f3 label %label learning rate %lr"
    export function trainPerceptron3(p: Perceptron, f1: number, f2: number, f3: number, label: number, lr: number) {
        p.train([f1, f2, f3], label, lr);
    }

    //% block="Perceptron %p predict %f1 %f2 %f3"
    export function perceptronPredict3(p: Perceptron, f1: number, f2: number, f3: number): number {
        return p.predict([f1, f2, f3]);
    }

    export class NeuralNet {
        inputSize: number;
        hiddenSize: number;
        outputSize: number;
        w1: number[][];
        b1: number[];
        w2: number[][];
        b2: number[];

        constructor(inputSize: number, hiddenSize: number, outputSize: number) {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;

            this.w1 = [];
            this.b1 = [];
            this.w2 = [];
            this.b2 = [];

            for (let i = 0; i < hiddenSize; i++) {
                let row: number[] = [];
                for (let j = 0; j < inputSize; j++) {
                    row.push(Math.randomRange(-10, 10) / 10);
                }
                this.w1.push(row);
                this.b1.push(0);
            }

            for (let i = 0; i < outputSize; i++) {
                let row: number[] = [];
                for (let j = 0; j < hiddenSize; j++) {
                    row.push(Math.randomRange(-10, 10) / 10);
                }
                this.w2.push(row);
                this.b2.push(0);
            }
        }

        private relu(x: number): number {
            return x > 0 ? x : 0;
        }

        private softmax(logits: number[]): number[] {
            let max = logits[0];
            for (let i = 1; i < logits.length; i++) {
                if (logits[i] > max) max = logits[i];
            }

            let exps: number[] = [];
            let sum = 0;

            for (let i = 0; i < logits.length; i++) {
                let e = Math.exp(logits[i] - max);
                exps.push(e);
                sum += e;
            }

            for (let i = 0; i < exps.length; i++) {
                exps[i] = exps[i] / sum;
            }

            return exps;
        }

        forward(features: number[]): number[] {
            let hidden: number[] = [];

            for (let i = 0; i < this.hiddenSize; i++) {
                let sum = this.b1[i];
                for (let j = 0; j < this.inputSize; j++) {
                    sum += this.w1[i][j] * features[j];
                }
                hidden.push(this.relu(sum));
            }

            let logits: number[] = [];

            for (let i = 0; i < this.outputSize; i++) {
                let sum = this.b2[i];
                for (let j = 0; j < this.hiddenSize; j++) {
                    sum += this.w2[i][j] * hidden[j];
                }
                logits.push(sum);
            }

            return this.softmax(logits);
        }

        predictClass(features: number[]): number {
            let probs = this.forward(features);
            let bestIdx = 0;
            let bestVal = probs[0];

            for (let i = 1; i < probs.length; i++) {
                if (probs[i] > bestVal) {
                    bestVal = probs[i];
                    bestIdx = i;
                }
            }

            return bestIdx;
        }

        savePayload(): string {
            let parts: string[] = [];
            parts.push("" + this.inputSize);
            parts.push("" + this.hiddenSize);
            parts.push("" + this.outputSize);

            for (let i = 0; i < this.hiddenSize; i++) {
                let row = this.w1[i];
                let line = "";
                for (let j = 0; j < this.inputSize; j++) {
                    if (j > 0) line = line + ",";
                    line = line + compressNumber(row[j]);
                }
                parts.push(line);
            }

            for (let i = 0; i < this.hiddenSize; i++) {
                parts.push(compressNumber(this.b1[i]));
            }

            for (let i = 0; i < this.outputSize; i++) {
                let row2 = this.w2[i];
                let line2 = "";
                for (let j = 0; j < this.hiddenSize; j++) {
                    if (j > 0) line2 = line2 + ",";
                    line2 = line2 + compressNumber(row2[j]);
                }
                parts.push(line2);
            }

            for (let i = 0; i < this.outputSize; i++) {
                parts.push(compressNumber(this.b2[i]));
            }

            return parts.join("|");
        }

        static loadPayload(payload: string): NeuralNet {
            let parts = payload.split("|");
            if (parts.length < 3) return new NeuralNet(3, 4, 2);
            let inp = parseInt(parts[0]);
            let hid = parseInt(parts[1]);
            let out = parseInt(parts[2]);
            let nn = new NeuralNet(inp, hid, out);

            let idx = 3;
            for (let i = 0; i < hid; i++) {
                let line = parts[idx++];
                let seg = line.split(",");
                for (let j = 0; j < inp; j++) {
                    nn.w1[i][j] = parseFloat(seg[j]);
                }
            }
            for (let i = 0; i < hid; i++) {
                nn.b1[i] = parseFloat(parts[idx++]);
            }
            for (let i = 0; i < out; i++) {
                let line2 = parts[idx++];
                let seg2 = line2.split(",");
                for (let j = 0; j < hid; j++) {
                    nn.w2[i][j] = parseFloat(seg2[j]);
                }
            }
            for (let i = 0; i < out; i++) {
                nn.b2[i] = parseFloat(parts[idx++]);
            }
            return nn;
        }
    }

    //% block="Create neural net inputs %inp hidden %hid outputs %out"
    export function createNeuralNet(inp: number, hid: number, out: number): NeuralNet {
        return new NeuralNet(inp, hid, out);
    }

    //% block="NN %nn predict class for %f1 %f2 %f3"
    export function nnPredict3(nn: NeuralNet, f1: number, f2: number, f3: number): number {
        return nn.predictClass([f1, f2, f3]);
    }

    export class DeepNet {
        inputSize: number;
        h1: number;
        h2: number;
        outputSize: number;
        w1: number[][];
        b1: number[];
        w2: number[][];
        b2: number[];
        w3: number[][];
        b3: number[];

        constructor(inputSize: number, h1: number, h2: number, outputSize: number) {
            this.inputSize = inputSize;
            this.h1 = h1;
            this.h2 = h2;
            this.outputSize = outputSize;

            this.w1 = [];
            this.b1 = [];
            this.w2 = [];
            this.b2 = [];
            this.w3 = [];
            this.b3 = [];

            for (let i = 0; i < h1; i++) {
                let row: number[] = [];
                for (let j = 0; j < inputSize; j++) {
                    row.push(Math.randomRange(-10, 10) / 10);
                }
                this.w1.push(row);
                this.b1.push(0);
            }

            for (let i = 0; i < h2; i++) {
                let row2: number[] = [];
                for (let j = 0; j < h1; j++) {
                    row2.push(Math.randomRange(-10, 10) / 10);
                }
                this.w2.push(row2);
                this.b2.push(0);
            }

            for (let i = 0; i < outputSize; i++) {
                let row3: number[] = [];
                for (let j = 0; j < h2; j++) {
                    row3.push(Math.randomRange(-10, 10) / 10);
                }
                this.w3.push(row3);
                this.b3.push(0);
            }
        }

        private relu(x: number): number {
            return x > 0 ? x : 0;
        }

        private softmax(logits: number[]): number[] {
            let max = logits[0];
            for (let i = 1; i < logits.length; i++) {
                if (logits[i] > max) max = logits[i];
            }
            let exps: number[] = [];
            let sum = 0;
            for (let i = 0; i < logits.length; i++) {
                let e = Math.exp(logits[i] - max);
                exps.push(e);
                sum += e;
            }
            for (let i = 0; i < exps.length; i++) {
                exps[i] = exps[i] / sum;
            }
            return exps;
        }

        forward(features: number[]): number[] {
            let h1v: number[] = [];
            for (let i = 0; i < this.h1; i++) {
                let sum = this.b1[i];
                for (let j = 0; j < this.inputSize; j++) {
                    sum += this.w1[i][j] * features[j];
                }
                h1v.push(this.relu(sum));
            }

            let h2v: number[] = [];
            for (let i = 0; i < this.h2; i++) {
                let sum = this.b2[i];
                for (let j = 0; j < this.h1; j++) {
                    sum += this.w2[i][j] * h1v[j];
                }
                h2v.push(this.relu(sum));
            }

            let logits: number[] = [];
            for (let i = 0; i < this.outputSize; i++) {
                let sum = this.b3[i];
                for (let j = 0; j < this.h2; j++) {
                    sum += this.w3[i][j] * h2v[j];
                }
                logits.push(sum);
            }

            return this.softmax(logits);
        }

        predictClass(features: number[]): number {
            let probs = this.forward(features);
            let bestIdx = 0;
            let bestVal = probs[0];
            for (let i = 1; i < probs.length; i++) {
                if (probs[i] > bestVal) {
                    bestVal = probs[i];
                    bestIdx = i;
                }
            }
            return bestIdx;
        }

        savePayload(): string {
            let parts: string[] = [];
            parts.push("" + this.inputSize);
            parts.push("" + this.h1);
            parts.push("" + this.h2);
            parts.push("" + this.outputSize);

            for (let i = 0; i < this.h1; i++) {
                let row = this.w1[i];
                let line = "";
                for (let j = 0; j < this.inputSize; j++) {
                    if (j > 0) line = line + ",";
                    line = line + compressNumber(row[j]);
                }
                parts.push(line);
            }

            for (let i = 0; i < this.h1; i++) {
                parts.push(compressNumber(this.b1[i]));
            }

            for (let i = 0; i < this.h2; i++) {
                let row2 = this.w2[i];
                let line2 = "";
                for (let j = 0; j < this.h1; j++) {
                    if (j > 0) line2 = line2 + ",";
                    line2 = line2 + compressNumber(row2[j]);
                }
                parts.push(line2);
            }

            for (let i = 0; i < this.h2; i++) {
                parts.push(compressNumber(this.b2[i]));
            }

            for (let i = 0; i < this.outputSize; i++) {
                let row3 = this.w3[i];
                let line3 = "";
                for (let j = 0; j < this.h2; j++) {
                    if (j > 0) line3 = line3 + ",";
                    line3 = line3 + compressNumber(row3[j]);
                }
                parts.push(line3);
            }

            for (let i = 0; i < this.outputSize; i++) {
                parts.push(compressNumber(this.b3[i]));
            }

            return parts.join("|");
        }

        static loadPayload(payload: string): DeepNet {
            let parts = payload.split("|");
            if (parts.length < 4) return new DeepNet(3, 6, 4, 3);

            let inp = parseInt(parts[0]);
            let h1 = parseInt(parts[1]);
            let h2 = parseInt(parts[2]);
            let out = parseInt(parts[3]);

            let dn = new DeepNet(inp, h1, h2, out);

            let idx = 4;

            for (let i = 0; i < h1; i++) {
                let seg = parts[idx++].split(",");
                for (let j = 0; j < inp; j++) {
                    dn.w1[i][j] = parseFloat(seg[j]);
                }
            }

            for (let i = 0; i < h1; i++) {
                dn.b1[i] = parseFloat(parts[idx++]);
            }

            for (let i = 0; i < h2; i++) {
                let seg2 = parts[idx++].split(",");
                for (let j = 0; j < h1; j++) {
                    dn.w2[i][j] = parseFloat(seg2[j]);
                }
            }

            for (let i = 0; i < h2; i++) {
                dn.b2[i] = parseFloat(parts[idx++]);
            }

            for (let i = 0; i < out; i++) {
                let seg3 = parts[idx++].split(",");
                for (let j = 0; j < h2; j++) {
                    dn.w3[i][j] = parseFloat(seg3[j]);
                }
            }

            for (let i = 0; i < out; i++) {
                dn.b3[i] = parseFloat(parts[idx++]);
            }

            return dn;
        }
    }

    //% block="Create deep net inputs %inp hidden1 %h1 hidden2 %h2 outputs %out"
    export function createDeepNet(inp: number, h1: number, h2: number, out: number): DeepNet {
        return new DeepNet(inp, h1, h2, out);
    }

    //% block="DeepNet %dn predict class for %f1 %f2 %f3"
    export function deepNetPredict3(dn: DeepNet, f1: number, f2: number, f3: number): number {
        return dn.predictClass([f1, f2, f3]);
    }

    //% block="Save model KNN %knn"
    export function saveModelKNN(knn: KNN): string {
        let type = "KNN";
        let payload = knn.savePayload();
        let full = type + "#" + payload;
        let cs = computeChecksum(full);
        return full + "#" + cs;
    }

    //% block="Save model NN %nn"
    export function saveModelNN(nn: NeuralNet): string {
        let type = "NN1";
        let payload = nn.savePayload();
        let full = type + "#" + payload;
        let cs = computeChecksum(full);
        return full + "#" + cs;
    }

    //% block="Save model DeepNet %dn"
    export function saveModelDeepNet(dn: DeepNet): string {
        let type = "DN2";
        let payload = dn.savePayload();
        let full = type + "#" + payload;
        let cs = computeChecksum(full);
        return full + "#" + cs;
    }

    //% block="Load model from %model"
    export function loadModel(model: string): any {
        let parts = model.split("#");
        if (parts.length != 3) return null;
        let type = parts[0];
        let payload = parts[1];
        let cs = parseInt(parts[2]);
        let full = type + "#" + payload;
        let check = computeChecksum(full);
        if (check != cs) {
            return null;
        }

        if (type == "KNN") {
            return KNN.loadPayload(payload);
        } else if (type == "NN1") {
            return NeuralNet.loadPayload(payload);
        } else if (type == "DN2") {
            return DeepNet.loadPayload(payload);
        }
        return null;
    }

    export class MultiDecisionNode {
        featureIndex: number;
        threshold: number;
        leftLabel: string;
        rightLabel: string;

        constructor(fi: number, th: number, left: string, right: string) {
            this.featureIndex = fi;
            this.threshold = th;
            this.leftLabel = left;
            this.rightLabel = right;
        }

        classify(features: number[]): string {
            if (features[this.featureIndex] < this.threshold) return this.leftLabel;
            return this.rightLabel;
        }
    }

    //% block="Train multi-class decision tree from %ds using feature %fi"
    export function trainMultiDecisionTree(ds: Dataset, fi: number): MultiDecisionNode {
        if (ds.samples.length == 0) {
            return new MultiDecisionNode(fi, 0, "A", "B");
        }

        let threshold = ds.samples[0][fi];

        let leftLabel = "left";
        let rightLabel = "right";

        let labels = ds.labels;
        if (labels.length >= 2) {
            leftLabel = labels[0];
            rightLabel = labels[1];
        }

        return new MultiDecisionNode(fi, threshold, leftLabel, rightLabel);
    }

    //% block="Multi decision tree %node classify %f1 %f2 %f3"
    export function multiDtClassify3(node: MultiDecisionNode, f1: number, f2: number, f3: number): string {
        return node.classify([f1, f2, f3]);
    }

    //% block="Keyword detected with threshold %t"
    export function keywordDetected(t: number): boolean {
        return input.soundLevel() > t;
    }

    //% block="Gesture shake detected"
    export function shakeDetected(): boolean {
        return input.isGesture(Gesture.Shake);
    }

    //% block="Gesture tilt left detected"
    export function tiltLeft(): boolean {
        return input.isGesture(Gesture.TiltLeft);
    }

    //% block="Is anomaly value %v mean %m tolerance %t"
    export function isAnomaly(v: number, m: number, t: number): boolean {
        return Math.abs(v - m) > t;
    }

    //% block="Record accel sample to %ds label %label on button A"
    export function recordAccelOnA(ds: Dataset, label: string) {
        input.onButtonPressed(Button.A, function () {
            let f = accelFeatures();
            ds.add(label, f);
            basic.showIcon(IconNames.Yes);
        });
    }

    //% block="Record accel sample to %ds label %label on button B"
    export function recordAccelOnB(ds: Dataset, label: string) {
        input.onButtonPressed(Button.B, function () {
            let f = accelFeatures();
            ds.add(label, f);
            basic.showIcon(IconNames.Yes);
        });
    }

    //% block="Record sound sample to %ds label %label on button A"
    export function recordSoundOnA(ds: Dataset, label: string) {
        input.onButtonPressed(Button.A, function () {
            let e = soundEnergy();
            ds.add(label, [e, 0, 0]);
            basic.showIcon(IconNames.Yes);
        });
    }

    //% block="Record sound sample to %ds label %label on button B"
    export function recordSoundOnB(ds: Dataset, label: string) {
        input.onButtonPressed(Button.B, function () {
            let e = soundEnergy();
            ds.add(label, [e, 0, 0]);
            basic.showIcon(IconNames.Yes);
        });
    }
}
