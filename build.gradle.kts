plugins {
    application
    java
}

val babylonRoot = System.getenv("JAVA_BABYLON_ROOT") ?: throw GradleException("JAVA_BABYLON_ROOT environment variable is not set")
val babylonHome = if (System.getProperty("os.name").contains("Mac")) {
    val macJDK = "$babylonRoot/Contents/Home"
    if (File(macJDK).exists()) macJDK else babylonRoot
} else {
    babylonRoot
}

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

tasks.register<Test>("integrationTest") {
    description = "Runs integration tests."
    group = "verification"
    
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath

    useJUnitPlatform {
        includeTags("integration")
    }
    jvmArgs(application.applicationDefaultJvmArgs)
    
    val tinyLlamaPath = System.getenv("TINY_LLAMA_PATH")
    if (tinyLlamaPath != null) {
        environment("TINY_LLAMA_PATH", tinyLlamaPath)
    }
    val llamaFp16Path = System.getenv("LLAMA_FP16_PATH")
    if (llamaFp16Path != null) {
        environment("LLAMA_FP16_PATH", llamaFp16Path)
    }
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
    mainClass.set("com.arturskowronski.llama3babylon.hat.GGUFReader")
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
