plugins {
    application
    java
}

val babylonHome = System.getenv("JAVA_BABYLON_HOME") ?: throw GradleException("JAVA_BABYLON_HOME environment variable is not set")

repositories {
    mavenCentral()
    flatDir {
        dirs("$babylonHome/hat/build")
    }
}

dependencies {
    implementation(files("$babylonHome/hat/build/hat-core-1.0.jar"))
    implementation(files("$babylonHome/hat/build/hat-optkl-1.0.jar"))
    implementation(files("$babylonHome/hat/build/hat-backend-java-seq-1.0.jar"))

    testImplementation("org.junit.jupiter:junit-jupiter:6.0.0")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.test {
    useJUnitPlatform()
    jvmArgs(application.applicationDefaultJvmArgs)
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(26))
    }
}

tasks.withType<JavaCompile> {
    options.compilerArgs.addAll(listOf(
        "--enable-preview",
        "--add-modules", "jdk.incubator.code",
        "--add-exports", "java.base/jdk.internal.vm.annotation=ALL-UNNAMED"
    ))
}

application {
    mainClass.set("com.arturskowronski.llama3babylon.hat.LlamaModel")
    applicationDefaultJvmArgs = listOf(
        "--enable-preview",
        "--add-modules=jdk.incubator.code",
        "--add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED",
        "-Djava.library.path=$babylonHome/hat/build"
    )
}

tasks.named<JavaExec>("run") {
    jvmArgs(application.applicationDefaultJvmArgs)
}
