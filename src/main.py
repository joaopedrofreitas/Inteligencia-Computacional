
from experiment import run_regression_experiment, run_classification_experiment


def main():
    print("Escolha o experimento:")
    print("1 - Regressão")
    print("2 - Classificação Multiclasse")
    
    while True:
        try:
            choice = input("\nDigite sua escolha (1 ou 2): ").strip()
            if choice == "1":
                run_regression_experiment()
                break
            elif choice == "2":
                run_classification_experiment()
                break
            else:
                print("Opção inválida. Digite 1 ou 2.")
        except KeyboardInterrupt:
            print("Interrompido pelo usuário.")
            break
        except Exception as e:
            print(f"Erro: {e}")
            break
    print("Concluído!")

if __name__ == "__main__":
    main()
