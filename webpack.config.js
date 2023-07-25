const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const CopyPlugin = require("copy-webpack-plugin");

const root = "./pipeliner/client";
module.exports = {
    entry: {
        index: `${root}/src/index.tsx`,
    },
    module: {
        rules: [
            {
                test: /\.tsx?$/,
                use: "ts-loader",
                exclude: /node_modules/,
            },
            {
                test: /\.s[ac]ss$/i,
                use: [
                    // Creates `style` nodes from JS strings
                    "style-loader",
                    // Translates CSS into CommonJS
                    "css-loader",
                    // Compiles Sass to CSS
                    "sass-loader",
                ],
            },
        ],
    },
    plugins: [
        new HtmlWebpackPlugin({
            title: "Pipeliner",
            template: `${root}/src/index.html`,
        }),
        new CopyPlugin({
            patterns: [{ from: `${root}/icons`, to: "icons" }],
        }),
    ],
    resolve: {
        extensions: [".tsx", ".ts", ".js"],
        modules: [path.resolve(__dirname, `${root}/src`), "node_modules"],
    },
    output: {
        filename: "[name].bundle.js",
        path: path.resolve(__dirname, `${root}/dist`),
        clean: true,
    },
    mode: "development",
    devServer: {
        static: `${root}`,
        port: 8001,
    },
};
