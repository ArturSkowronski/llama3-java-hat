plugins {
    application
    java
}

repositories {
    mavenCentral()
    flatDir {
        dirs("/Users/askowronski/GitHub/babylon/hat/build")
    }
}

dependencies {
    implementation(files("/Users/askowronski/GitHub/babylon/hat/build/hat-core-1.0.jar"))
    implementation(files("/Users/askowronski/GitHub/babylon/hat/build/hat-optkl-1.0.jar"))
    implementation(files("/Users/askowronski/GitHub/babylon/hat/build/hat-backend-java-seq-1.0.jar"))

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
    mainClass.set("com.arturskowronski.llama3babylon.hat.RuntimeCheck")
    applicationDefaultJvmArgs = listOf(
        "--enable-preview",
        "--add-modules=jdk.incubator.code",
        "--add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED",
        "-Djava.library.path=/Users/askowronski/GitHub/babylon/hat/build"
    )
}

tasks.named<JavaExec>("run") {
    jvmArgs(application.applicationDefaultJvmArgs)
}
